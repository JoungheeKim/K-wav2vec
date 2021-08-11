#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys

import editdistance
import numpy as np
import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging.meters import StopwatchMeter, TimeMeter
import unicodedata
from pyctcdecode import Alphabet, BeamSearchDecoderCTC
import multiprocessing
import argparse

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


## 평가를 위해서 ClovaCall Dictionary를 위해서 등록되지 않는 토큰을 넣어줍니다.(모델을 활용하여 토큰을 다 생성한 후에 target label에 대해서한 해당 토큰을 사용합니다.)
UNK_ADD_TOKENS = ['큼', '격', '칵', '톡', '난', '탕', '참', 'B', '꾸', '싼', '옥', '흡', '낫', '귀', '잃', '곧', '촬', '얘', '릇', '봤', '및', '푸', 'C', 'E', '균', '완', '밖', '앙', '놨', '씨', '헷', 'R', 'J', '억', '놀', '빠', '캡', '깎', '왜', '넷']
UNK_ADD_TOKENS += ['C', 'ᅤ', 'J', 'E', 'ᆶ', 'B', 'R']
UNK_ADD_TOKENS = list()

def add_asr_eval_argument():
    parser = argparse.ArgumentParser()
    ## 1. task
    parser.add_argument("--task", default='audio_pretraining', type=str, help="fairseq task")

    ## 2. decoding strategy
    parser.add_argument(
        "--decoder",
        choices=["beam", "max"],
        help="decoder",
    )

    ## 3. checkpoint_path
    parser.add_argument(
        "--checkpoint-path",
        help="checkpoint paths",
    )

    ## 4. subset
    parser.add_argument(
        "--gen-subset",
        help="choose subset name",
    )

    ## 5. results-path
    parser.add_argument(
        "--results-path",
        help="log path",
    )

    ## 6. criterion
    parser.add_argument(
        "--criterion",
        choices=["ctc", "multi_ctc"],
        help="choose criterion",
    )

    ## 7. labels
    parser.add_argument(
        "--labels",
        default='ltr',
        help="label file name",
    )

    parser.add_argument(
        "--post-process",
        default='letter',
        help="label file name",
    )

    parser.add_argument(
        "--add-weight",
        type=float,
        default=0.5,
        help="contribution weights for multi task model(single model use default:0.5)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="batch size per gpu",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000000,
        help="max-tokens",
    )

    parser.add_argument(
        "--experiments-dir",
        type=str,
        default='/code/gitRepo/wav2vec_exp/experiments/experiments.csv',
        help="if present, loads emissions from this file",
    )
    parser.add_argument(
        "--additional-output",
        action="store_true",
        help="if present, loads emissions from this file",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="if ture, cpu is runnning instead gpu",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="if you want to use fp16, check option",
    )

    parser.add_argument(
        "--log-format",
        default="tqdm",
        help="log format",
    )

    parser.add_argument(
        "--log-interval",
        default=1,
        help="log interval",
    )
    return parser


def get_dataset_itr(args, task, models):
    return task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=False,
        #num_shards=args.num_shards,
        #shard_id=args.shard_id,
        #num_workers=args.num_workers,
        #data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)


def sum_log_scores(s1: float, s2: float) -> float:
    """Sum log odds in a numerically stable way."""
    # this is slightly faster than using max
    if s1 >= s2:
        log_sum = s1 + math.log(1 + math.exp(s2 - s1))
    else:
        log_sum = s2 + math.log(1 + math.exp(s1 - s2))
    return log_sum


def process_predictions(
        args, hypos, sp, tgt_dict, target_tokens, res_files, speaker, id
):
    errs_wer = 0
    lengths_wer = 0

    errs_swer = 0
    lengths_swer = 0

    errs_cer = 0
    lengths_cer = 0

    hyp_pieces, tgt_pieces, hyp_words, tgt_words = [], [], [], []
    count = 0
    for hypo in hypos[: min(len(hypos), args.nbest)]:
        hyp_piece = hypo["tokens"]

        if "words" in hypo:
            hyp_word = " ".join(hypo["words"])
        else:
            hyp_word = post_process(hyp_piece, args.post_process)

        if res_files is not None:
            print(
                "{} ({}-{})".format(" ".join(hyp_piece.replace(' ',"|")+"|"), speaker, id),
                file=res_files["hypo.units"],
            )
            print(
                "{} ({}-{})".format(hyp_word, speaker, id),
                file=res_files["hypo.words"],
            )

        tgt_piece = tgt_dict.string(target_tokens)
        tgt_word = post_process(tgt_piece, args.post_process)

        if res_files is not None:
            print(
                "{} ({}-{})".format(tgt_piece, speaker, id),
                file=res_files["ref.units"],
            )
            print(
                "{} ({}-{})".format(tgt_word, speaker, id), file=res_files["ref.words"]
            )
            # only score top hypothesis


        ## normalize(한글 NFKD 일 경우 글자 갯수가 다르게 측정되는 걸 방지하기 위하여)
        hyp_word = unicodedata.normalize("NFC", hyp_word)
        tgt_word = unicodedata.normalize("NFC", tgt_word)

        hyp_word=" ".join(hyp_word.split())
        tgt_word=" ".join(tgt_word.split())

        err_w, length_w = editdistance.eval(hyp_word.split(), tgt_word.split()), len(tgt_word.split())
        err_c, length_c = editdistance.eval(hyp_word, tgt_word), len(tgt_word)

        ##
        norm_hyp_word, norm_tgt_word = get_norm_text(hyp_word, tgt_word)
        err_sw, length_sw = editdistance.eval(norm_hyp_word.split(), norm_tgt_word.split()), len(norm_tgt_word.split())

        errs_wer += err_w
        lengths_wer += length_w
        errs_cer += err_c
        lengths_cer += length_c
        errs_swer += err_sw
        lengths_swer += length_sw

        hyp_pieces.append(hyp_piece)
        tgt_pieces.append(tgt_piece)
        hyp_words.append(hyp_word)
        tgt_words.append(tgt_word)
        count += 1
        assert count < 2, "여기로 오면 안되는데 한개만 뽑아야 되는데? 원래코드라면"

    return {
        'errs_wer': errs_wer,
        'lengths_wer': lengths_wer,
        'errs_cer': errs_cer,
        'lengths_cer': lengths_cer,
        'errs_swer': errs_swer,
        'lengths_swer': lengths_swer,
    }


def prepare_result_files(args):
    def get_res_file(file_prefix):
        path = os.path.join(
            args.results_path,
            "{}-{}-{}.txt".format(
                file_prefix, os.path.basename(args.checkpoint_path), args.gen_subset
            ),
        )
        return open(path, "w", buffering=1)

    if not args.results_path:
        return None

    return {
        "hypo.words": get_res_file("hypo.word"),
        "hypo.units": get_res_file("hypo.units"),
        "ref.words": get_res_file("ref.word"),
        "ref.units": get_res_file("ref.units"),
    }


def optimize_models(args, use_cuda, models):
    """Optimize ensemble for generation"""
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()


class ExistingEmissionsDecoder(object):
    def __init__(self, decoder, emissions):
        self.decoder = decoder
        self.emissions = emissions

    def generate(self, models, sample, **unused):
        ids = sample["id"].cpu().numpy()
        try:
            emissions = np.stack(self.emissions[ids])
        except:
            print([x.shape for x in self.emissions[ids]])
            raise Exception("invalid sizes")
        emissions = torch.from_numpy(emissions)
        return self.decoder.decode(emissions)

class BeamDecoder(object):
    def __init__(self, args, tgt_dict, add_tgt_dict):
        self.args = args
        self.criterion_type= None
        if args.criterion in ["ctc", 'multi_ctc']:
            self.criterion_type = "ctc"
            self.space_token='|'

        self.tgt_dict = tgt_dict
        self.add_tgt_dict = add_tgt_dict
        self.beam = args.beam


        ## portion
        #self.add_weight = np.log(1.0)
        self.origin_weight = np.log(1.0)

        # make alphabet
        vocab_list = list(tgt_dict.symbols)
        blank_idx = tgt_dict.index(self.space_token)

        # convert ctc blank character representation
        vocab_list[0] = ""
        # replace special characters
        vocab_list[1] = "⁇"
        vocab_list[2] = "⁇"
        vocab_list[3] = "⁇"
        # convert space character representation
        vocab_list[blank_idx] = " "
        # specify ctc blank char index, since conventionally it is the last entry of the logit matrix
        alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=0)

        # build the decoder and decode the logits
        self.decoder = BeamSearchDecoderCTC(alphabet)
        self.add_decoder = None
        if add_tgt_dict is not None:
            self.add_weight = np.log(args.add_weight)
            self.origin_weight = np.log(1.0 - args.add_weight)
            print("add decoder 추가")
            # make alphabet
            add_vocab_list = list(add_tgt_dict.symbols)
            add_blank_idx = add_tgt_dict.index(self.space_token)
            # convert ctc blank character representation
            add_vocab_list[0] = ""
            # replace special characters
            add_vocab_list[1] = "⁇"
            add_vocab_list[2] = "⁇"
            add_vocab_list[3] = "⁇"
            # convert space character representation
            add_vocab_list[add_blank_idx] = " "
            # specify ctc blank char index, since conventionally it is the last entry of the logit matrix
            add_alphabet = Alphabet.build_alphabet(add_vocab_list, ctc_token_idx=0)

            # build the decoder and decode the logits
            self.add_decoder = BeamSearchDecoderCTC(add_alphabet)
            
    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        model = models[0]
        encoder_out = model(**encoder_input)

        emissions=None
        add_emissions=None
        if self.criterion_type == "ctc":
            if hasattr(model, "get_logits"):
                emissions = model.get_logits(encoder_out) # no need to normalize emissions
                emissions = emissions.transpose(0, 1).float().cpu().detach().numpy()
                if self.add_decoder is not None:
                    add_emissions = model.get_addlogits(encoder_out) # no need to normalize emissions
                    add_emissions = add_emissions.transpose(0, 1).float().cpu().detach().numpy()
            else:
                emissions = model.get_normalized_probs(encoder_out, log_probs=True)
                emissions = emissions.transpose(0, 1).float().cpu().detach().numpy()
                if self.add_decoder is not None:
                    add_emissions = model.get_normalized_addprobs(encoder_out, log_probs=True)
                    add_emissions = add_emissions.transpose(0, 1).float().cpu().detach().numpy()
        
        return emissions, add_emissions

    def decode_batch(self, emissions, add_emissions):
        if add_emissions is None:
            add_emissions = emissions
        
        with multiprocessing.Pool() as pool:
            beam_outputs_list = self.decoder.decode_beams_batch(pool, emissions, beam_width=self.beam)
        results = beam_outputs_list


        if self.add_decoder is not None:
            results = list()
            with multiprocessing.Pool() as pool:
                add_beam_outputs_list = self.add_decoder.decode_beams_batch(pool, add_emissions, beam_width=self.beam)

            for beam_outputs, add_beam_outputs in zip(beam_outputs_list, add_beam_outputs_list):
                temp_results = dict()
                for beam_output in beam_outputs:
                    
                    temp_sent = unicodedata.normalize("NFC", beam_output[0])
                    temp_score = beam_output[-2]
                    
                    if temp_sent in temp_results:
                        temp_results[temp_sent] = sum_log_scores((self.origin_weight+temp_score), temp_results[temp_sent])
                    else:
                        temp_results[temp_sent] = (self.origin_weight+temp_score)
                        
                for beam_output in add_beam_outputs:
                    temp_sent = unicodedata.normalize("NFC", beam_output[0])
                    temp_score = beam_output[-2]

                    if temp_sent in temp_results:
                        temp_results[temp_sent] = sum_log_scores((self.add_weight+temp_score), temp_results[temp_sent])
                    else:
                        temp_results[temp_sent] = (self.add_weight+temp_score)

                ## Sort
                temp_results = sorted(temp_results.items(), key=lambda item: item[1], reverse=True)
                results.append(temp_results)

        results = [[{
                "tokens": result[0][0],
                "words": result[0][0].split(),
                "score" : result[0][-1]
            }] for result in results]
        
        return results 


    def decode(self, emissions, add_emissions):
        if add_emissions is None:
            add_emissions = emissions

        results = list()
        for emission, add_emission in zip(emissions, add_emissions):
            beam_outputs = self.decoder.decode_beams(emission, beam_width=self.beam)
            temp_results = beam_outputs
            if self.add_decoder is not None:
                add_beam_outputs = self.add_decoder.decode_beams(add_emission, beam_width=self.beam)

                temp_results = dict()
                for beam_output in beam_outputs:
                    temp_sent = unicodedata.normalize("NFC", beam_output[0])
                    temp_score = beam_output[-2]

                    if temp_sent in temp_results:
                        temp_results[temp_sent] = sum_log_scores((self.origin_weight+temp_score), temp_results[temp_sent])
                    else:
                        temp_results[temp_sent] = (self.origin_weight+temp_score)
                        
                
                for beam_output in add_beam_outputs:
                    temp_sent = unicodedata.normalize("NFC", beam_output[0])
                    temp_score = beam_output[-2]

                    if temp_sent in temp_results:
                        temp_results[temp_sent] = sum_log_scores((self.add_weight+temp_score), temp_results[temp_sent])
                    else:
                        temp_results[temp_sent] = (self.add_weight+temp_score)

                ## Sort
                temp_results = sorted(temp_results.items(), key=lambda item: item[1], reverse=True)
                #assert temp_results[0][0] == unicodedata.normalize("NFC", beam_outputs[0][0]), print(temp_results[0][-1], beam_outputs[0][-1])

            results.append([{
                "tokens": temp_results[0][0],
                "words": temp_results[0][0].split(),
                "score" : temp_results[0][-1]
            }])    

        return results 



    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions, add_emissions = self.get_emissions(models, encoder_input)
        
        #return self.decode_batch(emissions, add_emissions)
        return self.decode(emissions, add_emissions)

class BeamMaxDecoder(BeamDecoder):
    def __init__(self, args, tgt_dict, add_tgt_dict):
        super().__init__(args, tgt_dict, add_tgt_dict)
    
    def decode_batch(self, emissions, add_emissions):
        if add_emissions is None:
            add_emissions = emissions
        
        with multiprocessing.Pool() as pool:
            beam_outputs_list = self.decoder.decode_beams_batch(pool, emissions, beam_width=self.beam)
        results = beam_outputs_list


        if self.add_decoder is not None:
            results = list()
            with multiprocessing.Pool() as pool:
                add_beam_outputs_list = self.add_decoder.decode_beams_batch(pool, add_emissions, beam_width=self.beam)

            for beam_outputs, add_beam_outputs in zip(beam_outputs_list, add_beam_outputs_list):
                temp_results = dict()
                for beam_output in beam_outputs:
                    
                    temp_sent = unicodedata.normalize("NFC", beam_output[0])
                    temp_score = beam_output[-2]
                    
                    if temp_sent in temp_results:
                        temp_results[temp_sent] = max((self.origin_weight+temp_score), temp_results[temp_sent])
                    else:
                        temp_results[temp_sent] = (self.origin_weight+temp_score)
                        
                for beam_output in add_beam_outputs:
                    temp_sent = unicodedata.normalize("NFC", beam_output[0])
                    temp_score = beam_output[-2]

                    if temp_sent in temp_results:
                        temp_results[temp_sent] = max((self.add_weight+temp_score), temp_results[temp_sent])
                    else:
                        temp_results[temp_sent] = (self.add_weight+temp_score)

                ## Sort
                temp_results = sorted(temp_results.items(), key=lambda item: item[1], reverse=True)
                results.append(temp_results)

        results = [[{
                "tokens": result[0][0],
                "words": result[0][0].split(),
                "score" : result[0][-1]
            }] for result in results]
        
        return results 


    


space_sym = " "
unmatched_sym = "<u>"


def get_score(a, b):
    # get score for Levenshtein
    if a == b:
        return 0
    else:
        return 1


def norm_space(token):
    # get normalized token
    return token.replace(space_sym, "")


## KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition, 2020 논문에서 제시한 wser 방법 참조
## https://github.com/hchung12/espnet/blob/master/egs/ksponspeech/asr1/local/get_space_normalized_hyps.py
def get_norm_text(hyps, refs):
    # this implementation is modified from LevenshteinAlignment of the kaldi toolkit
    # - https://github.com/kaldi-asr/kaldi/blob/master/src/bin/align-text.cc

    # initialize variables
    hyp_norm, ref_norm = [], []

    # length of two sequences
    hlen, rlen = len(hyps), len(refs)

    # initialization
    # - this is very memory-inefficiently implemented using a vector of vectors
    scores = np.zeros((hlen + 1, rlen + 1))
    for r in range(0, rlen + 1):
        scores[0][r] = r
    for h in range(1, hlen + 1):
        scores[h][0] = scores[h - 1][0] + 1
        for r in range(1, rlen + 1):
            hyp_nosp, ref_nosp = norm_space(hyps[h - 1]), norm_space(refs[r - 1])
            sub_or_cor = scores[h - 1][r - 1] + get_score(hyp_nosp, ref_nosp)
            insert, delete = scores[h - 1][r] + 1, scores[h][r - 1] + 1
            scores[h][r] = min(sub_or_cor, insert, delete)

    # traceback and compute the alignment
    h, r = hlen, rlen  # start from the bottom
    while h > 0 or r > 0:
        if h == 0:
            last_h, last_r = h, r - 1
        elif r == 0:
            last_h, last_r = h - 1, r
        else:
            # get score
            hyp_nosp, ref_nosp = norm_space(hyps[h - 1]), norm_space(refs[r - 1])
            sub_or_cor = scores[h - 1][r - 1] + get_score(hyp_nosp, ref_nosp)
            insert, delete = scores[h - 1][r] + 1, scores[h][r - 1] + 1

            # choose sub_or_cor if all else equal
            if sub_or_cor <= min(insert, delete):
                last_h = h - 1
                last_r = r - 1
            else:
                if insert < delete:
                    last_h = h - 1
                    last_r = r
                else:
                    last_h = h
                    last_r = r - 1

        c_hyp = hyps[last_h] if last_h != h else ""
        c_ref = refs[last_r] if last_r != r else ""
        h, r = last_h, last_r

        # do word-spacing normalization
        if c_hyp != c_ref and norm_space(c_hyp) == norm_space(c_ref):
            c_hyp = c_ref
        if c_hyp != "":
            hyp_norm.append(c_hyp)
        if c_ref != "":
            ref_norm.append(c_ref)

    # reverse list
    hyp_norm.reverse()
    ref_norm.reverse()

    return ("".join(hyp_norm), "".join(ref_norm))


def main(args):

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 4000000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    logger.info("| decoding with criterion {}".format(args.criterion))

    task = tasks.setup_task(args)

    # Set dictionary
    import copy
    tgt_dict = copy.deepcopy(task.target_dictionary)
    add_tgt_dict = None
    
    # 실험용
    if hasattr(task, "additional_dictionary") and task.additional_dictionary is not None:
        add_tgt_dict = copy.deepcopy(task.additional_dictionary)


    logger.info("| loading model(s) from {}".format(args.checkpoint_path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.checkpoint_path),
        #arg_overrides=ast.literal_eval(args.model_overrides),
        task=task,
        #suffix=args.checkpoint_suffix,
        #strict=(args.checkpoint_shard_count == 1),
        #num_shards=args.checkpoint_shard_count,
        state=None,
    )

    optimize_models(args, use_cuda, models)

    ## 데이터 불러오기 전 세팅
    for token in UNK_ADD_TOKENS:
        task.target_dictionary.add_symbol(token)
    if hasattr(task, "additional_dictionary") and task.additional_dictionary is not None:
        for token in UNK_ADD_TOKENS:
            task.additional_dictionary.add_symbol(token)

    task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)
    
    
    logger.info(
        "| {} {} {} examples".format(
            args.data, args.gen_subset, len(task.dataset(args.gen_subset))
        )
    )

    # hack to pass transitions to W2lDecoder
    if args.criterion == "asg_loss":
        raise NotImplementedError("asg_loss is currently not supported")
        # trans = criterions[0].asg.trans.data
        # args.asg_transitions = torch.flatten(trans).tolist()

    # Load dataset (possibly sharded)
    itr = get_dataset_itr(args, task, models)

    # Initialize generator
    gen_timer = StopwatchMeter()

    def build_generator(args, tgt_dict, add_tgt_dict):
        w2l_decoder = getattr(args, "decoder", None)
        if w2l_decoder == "beam":
            return BeamDecoder(args, tgt_dict, add_tgt_dict)
        elif w2l_decoder == "max":
            return BeamMaxDecoder(args, tgt_dict, add_tgt_dict)

        raise NotImplementedError("nothing is selected for decoding")

    # please do not touch this unless you test both generate.py and infer.py with audio_pretraining task
    generator = build_generator(args, tgt_dict, add_tgt_dict)

    num_sentences = 0

    if args.results_path is not None and not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    max_source_pos = (
        utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        ),
    )

    if max_source_pos is not None:
        max_source_pos = max_source_pos[0]
        if max_source_pos is not None:
            max_source_pos = max_source_pos[0] - 1

    res_files = prepare_result_files(args)

    errs_wer = 0
    lengths_wer = 0

    errs_swer = 0
    lengths_swer = 0

    errs_cer = 0
    lengths_cer = 0

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if "net_input" not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:

                ## 여기 추가
                if args.additional_output:
                    prefix_tokens = sample["add_target"][:, : args.prefix_size]
                else:
                    prefix_tokens = sample["target"][:, : args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample["id"].tolist()):
                speaker = None
                # id = task.dataset(args.gen_subset).ids[int(sample_id)]
                id = sample_id
                ## 여기 추가
                if args.additional_output:
                    toks = (
                    sample["add_target"][i, :]
                    if "add_target_label" not in sample
                    else sample["add_target_label"][i, :]
                )
                else:
                    toks = (
                        sample["target"][i, :]
                        if "target_label" not in sample
                        else sample["target_label"][i, :]
                    )
                target_tokens = utils.strip_pad(toks, tgt_dict.pad()).int().cpu()
                # Process top predictions
                postprocess_result = process_predictions(
                    args,
                    hypos[i],
                    None,
                    #tgt_dict,
                    task.target_dictionary,
                    target_tokens,
                    res_files,
                    speaker,
                    id,
                )

                errs_wer += postprocess_result['errs_wer']
                lengths_wer += postprocess_result['lengths_wer']
                errs_cer += postprocess_result['errs_cer']
                lengths_cer += postprocess_result['lengths_cer']
                errs_swer += postprocess_result['errs_swer']
                lengths_swer += postprocess_result['lengths_swer']

            wps_meter.update(num_generated_tokens)
            t.log({"wps": round(wps_meter.avg)})
            num_sentences += (
                sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
            )

    wer = None
    cer = None
    swer = None

    if lengths_wer > 0:
        wer = errs_wer * 100.0 / lengths_wer
        logger.info(f"WER: {wer}")

    if lengths_cer > 0:
        cer = errs_cer * 100.0 / lengths_cer
        logger.info(f"CER: {cer}")

    if lengths_swer > 0:
        swer = errs_swer * 100.0 / lengths_swer
        logger.info(f"sWER: {swer}")

    logger.info(
        "| Processed {} sentences ({} tokens) in {:.1f}s ({:.2f}"
        "sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    logger.info("| Generate {} with beam={}".format(args.gen_subset, args.beam))
    return task, wer, cer, swer


def make_parser():
    parser = add_asr_eval_argument()
    return parser


def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    task, wer, cer, swer = main(args)

    writer = ResultWriter(args.experiments_dir)
    results = {
        'cer': cer,
        'wer': wer,
        'swer': swer,
    }
    writer.update(args, **results)

    return wer, cer, swer


from datetime import datetime
import pandas as pd


class ResultWriter:
    def __init__(self, directory):

        self.dir = directory
        self.hparams = None
        self.load()
        self.writer = dict()

    def remove_list(self):
        remove_list = []
        for key, item in self.writer.items():
            if type(item) == list:
                remove_list.append(key)
            elif item == None or item == '':
                remove_list.append(key)

        for key in remove_list:
            self.writer.pop(key)

    def update(self, args, **results):
        now = datetime.now()
        date = "%s-%s %s:%s" % (now.month, now.day, now.hour, now.minute)
        self.writer.update({"date": date})
        self.writer.update(results)
        self.writer.update(vars(args))
        self.remove_list()

        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = self.hparams.append(self.writer, ignore_index=True)
        self.save()

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.makedirs(path)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None


if __name__ == "__main__":
    wer, cer, swer = cli_main()
    print(wer, cer, swer, end='')
