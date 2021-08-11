import argparse
from preprocess import preprocess
import os
from pathlib import Path
import wave
import numpy as np
import unicodedata
import random
from tqdm import tqdm
import re
import yaml
import sys
import librosa

## Fairseq 스타일로 변환하기
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default='/code/gitRepo/data/aihub/ksponspeech', metavar="DIR",
        help="root directory containing flac files to index"
    )

    parser.add_argument(
        "--info", default=None, metavar="DIR",
        help="전처리 추가적으로 수행한 것."
    )
    parser.add_argument(
        "--do_info", action="store_true",
        help="전처리 추가적으로 수행할지 여부 확인"
    )
    parser.add_argument(
        "--do_remove", action="store_true",
        help="한글 음소가 아닌 숫자, 영어가 포함되어 있는 모든 단어를 삭제할지 여부 확인"
    )
    parser.add_argument(
        "--token_limit", default=sys.maxsize, type=int,
        help="최대 글자수 체크"
    )

    parser.add_argument(
        "--dest", default='manifest_temp', type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="pcm", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument('--preprocess_mode', type=str,
                        default='phonetic',
                        help='Ex) (70%)/(칠 십 퍼센트) 확률이라니 (뭐 뭔)/(모 몬) 소리야 진짜 (100%)/(백 프로)가 왜 안돼?'
                             'phonetic: 칠 십 퍼센트 확률이라니 모 몬 소리야 진짜 백 프로가 왜 안돼?'
                             'spelling: 70% 확률이라니 뭐 뭔 소리야 진짜 100%가 왜 안돼?')
    parser.add_argument('--output_unit', type=str,
                        default='grapheme',
                        help='character or subword or grapheme')
    parser.add_argument('--additional_output_unit', type=str,
                        default=None,
                        help='character or subword or grapheme')
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--time",
        default=None,
        type=str,
        metavar="MIN",
        help="set if you want make split manifest",
    )
    parser.add_argument('--script_path', type=str,
                        default="/code/gitRepo/data/aihub/ksponspeech/KsponSpeech_scripts",
                        help='AIHUB에서 제공해 주는 스크립트 폴더')
    parser.add_argument(
        "--del_silence", action="store_true",
        help="음성이 없는 곳을 삭제하는 건 어때?"
    )
    return parser


def find_index(durations, limit):
    for idx in range(len(durations)):
        if sum(durations[:idx]) > limit:
            return idx
    return len(durations)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(yaml_path):
    # Read YAML file
    with open(yaml_path, 'r') as stream:
        data_loaded = yaml.load(stream, Loader=yaml.FullLoader)
    return data_loaded


def load_info(info_path):
    if not os.path.isdir(info_path):
        return {}

    info_files = [filename for filename in os.listdir(info_path) if '.yaml' in filename]
    info_data = {}
    for filename in info_files:
        file_path = os.path.join(info_path, filename)
        temp_data = load_yaml(file_path)
        info_data.update(temp_data)
    return info_data

def save_converted_info(args, name, converted_info):
    if len(converted_info) == 0:
        return

    yaml_dict = {k: v for k, v in sorted(converted_info.items(), key=lambda item: (len(item[0]), item[0]))}
    with open(os.path.join(args.dest, '{}.yaml'.format(name)), 'w', encoding="utf-8") as write_f:
        yaml.dump(yaml_dict, write_f, allow_unicode=True, default_style=None, default_flow_style=False)


def save_wrong_script(args, name, transcripts, fileinfo, raw_sentences, new_sentences):
    ## 틀린 것 저장하기

    ## 알파벳 추가
    reg = re.compile(r'[A-Z]')
    yaml_dict = {}
    for grapheme_transcript, fileitem, raw_sentence, new_sentence in zip(transcripts, fileinfo, raw_sentences,
                                                                         new_sentences):
        graphemes = grapheme_transcript.split()
        file_num = Path(fileitem.split()[0]).stem.split("_")[1]
        assert len(file_num) == 6

        for grapheme in graphemes:
            if grapheme.isdigit() or reg.match(grapheme):
                yaml_dict[file_num] = str(raw_sentence.replace('\n', ''))

    if len(yaml_dict) == 0:
        return

    ## Sorting
    yaml_dict = {k: v for k, v in sorted(yaml_dict.items(), key=lambda item: (len(item[0]), item[0]))}
    with open(os.path.join(args.dest, '{}.yaml'.format(name)), 'w', encoding="utf-8") as write_f:
        yaml.dump(yaml_dict, write_f, allow_unicode=True, default_style=None, default_flow_style=False)


def save_dict(args, transcripts, dict_name='dict.ltr.txt', alphabet_name='alphabet.txt'):
    vocab_list = list()
    vocab_freq = list()
    for grapheme_transcript in transcripts:
        graphemes = grapheme_transcript.split()

        for grapheme in graphemes:
            if grapheme not in vocab_list:
                vocab_list.append(grapheme)
                vocab_freq.append(1)
            else:
                vocab_freq[vocab_list.index(grapheme)] += 1

    ## write ltr
    vocab_freq, vocab_list = zip(*sorted(zip(vocab_freq, vocab_list), reverse=True))
    with open(os.path.join(args.dest, dict_name), 'w') as write_f:
        for idx, (grpm, freq) in enumerate(zip(vocab_list, vocab_freq)):
            print("{} {}".format(grpm, freq), file=write_f)

    ## Write Vocab files
    with open(os.path.join(args.dest, alphabet_name), 'w', encoding='UTF8') as write_f:
        print("# Each line in this file represents the Unicode codepoint (UTF-8 encoded)", file=write_f)
        print("# associated with a numeric label.", file=write_f)
        print("# A line that starts with # is a comment. You can escape it with \# if you wish", file=write_f)
        print("# to use '#' as a label.", file=write_f)
        for token in vocab_list:
            print(token, file=write_f)
        ## final token must be \n
        print('', file=write_f)

        print("# The last (non-comment) line needs to end with a newline.", file=write_f, end='')

    return

def save_lexicon(args, texts, lexicon_name='lexicon.lst'):
    vocab_list = {}
    for text in texts:
        for word in text.split():
            new_word = word + "|"
            vocab_list[word] = " ".join(new_word)

    ## Write Vocab files
    ## Sorting
    vocab_list = {k: v for k, v in sorted(vocab_list.items(), key=lambda item: item[0])}
    with open(os.path.join(args.dest, lexicon_name), 'w', encoding='UTF8') as write_f:
        for k, v in vocab_list.items():
            print("{}\t{}".format(k,v), file=write_f)
    return




def save_files(args, file_name, dir_path, fileinfo, texts, transcripts):
    with open(os.path.join(args.dest, file_name + ".tsv"), 'w') as tsv_out, open(
            os.path.join(args.dest, file_name + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.dest, file_name + ".wrd"), "w"
    ) as wrd_out:

        print(dir_path, file=tsv_out)
        for tsv_item, wrd_item, ltr_item in zip(fileinfo, texts, transcripts):
            print(tsv_item, file=tsv_out)
            print(wrd_item, file=wrd_out)
            print(ltr_item + " |", file=ltr_out)

    print("save files [{}]".format(file_name))
    return



def pcm2wav(pcm_file, channels=1, bit_depth=16, sampling_rate=16000):
    wav_file = str(Path(pcm_file).with_suffix('.wav'))
    # Check if the options are valid.
    if bit_depth % 8 != 0:
        raise ValueError("bit_depth " + str(bit_depth) + " must be a multiple of 8.")

    # Read the .pcm file as a binary file and store the data to pcm_data
    with open(pcm_file, 'rb') as opened_pcm_file:
        pcm_data = opened_pcm_file.read()
        with wave.open(wav_file, 'wb') as obj2write:
            obj2write.setnchannels(channels)
            obj2write.setsampwidth(bit_depth // 8)
            obj2write.setframerate(sampling_rate)
            obj2write.writeframes(pcm_data)

    return wav_file

def load_script(args, script_path, info_data, token_limit=sys.maxsize):
    assert os.path.isfile(script_path)

    fileinfo = list()
    durations = list()
    texts = list()
    audio_nums = list()
    transcripts = list()

    additional_texts = list()
    additional_transcripts = list()

    raw_sentences = list()
    new_sentences = list()

    converted_info = {}

    reg = re.compile(r'.*[a-zA-Z0-9]')
    limit_count = 0
    remove_count = 0
    with open(script_path, "r") as f:
        for line in tqdm(f):
            convert_flag = False

            items = line.split(" :: ")
            file_path = os.path.join(args.root, items[0])
            file_path = os.path.realpath(file_path)
            audio_num = str(Path(file_path).stem.split("_")[1])
            raw_sentence = items[1]
            if len(audio_num) ==6 and audio_num in info_data:
                raw_sentence = info_data[audio_num]
                convert_flag=True

            ## 확장자 확인
            if args.ext == 'pcm':
                try:
                    wav = np.memmap(file_path, dtype='h', mode='r').astype('float32') / 32767
                    sr = 16000
                except ValueError:
                    # print('pcm load 에러 wave로 교체 [{}]'.format(file_path))
                    file_path = pcm2wav(file_path)
                    wav, sr = librosa.load(file_path, sr=16000)

            elif args.ext in ['flac', 'wav']:
                wav, sr = librosa.load(file_path, sr=16000)
            else:
                raise ValueError("Unsupported extention method : {0}".format(args.ext))

            if args.del_silence:
                non_silence_indices = librosa.effects.split(wav, top_db=30)
                wav = np.concatenate([wav[start:end] for start, end in non_silence_indices])
            frames = len(wav)

            if len(audio_num) ==6:
                new_sentence = preprocess(raw_sentence=raw_sentence, mode=args.preprocess_mode, audio_num=audio_num)
            else:
                new_sentence = raw_sentence.replace('\n', '')

            ##################################
            if len(new_sentence) > token_limit:
                limit_count+=1
                continue

            if args.do_remove and reg.match(new_sentence) and args.preprocess_mode != 'spelling':
                converted_info[audio_num] = new_sentence
                remove_count += 1
                continue
            #################################


            ## 저장 모드는 여기에 추가하기.
            if args.output_unit == 'grapheme':
                texts.append(unicodedata.normalize('NFKD', new_sentence).upper())
                transcripts.append(" ".join(unicodedata.normalize('NFKD', new_sentence).replace(' ', '|')).upper())
            elif args.output_unit == 'character':
                texts.append(new_sentence.upper())
                transcripts.append(" ".join(list(new_sentence.replace(' ', '|').upper())))
            else:
                raise ValueError("Unsupported preprocess method : {0}".format(args.output_unit))

            ## 저장 모드는 여기에 추가하기.
            if args.additional_output_unit is not None:
                if args.additional_output_unit == 'grapheme':
                    additional_texts.append(unicodedata.normalize('NFKD', new_sentence).upper())
                    additional_transcripts.append(" ".join(unicodedata.normalize('NFKD', new_sentence).replace(' ', '|')).upper())
                elif args.additional_output_unit == 'character':
                    additional_texts.append(new_sentence.upper())
                    additional_transcripts.append(" ".join(list(new_sentence.replace(' ', '|').upper())))
                else:
                    raise ValueError("Unsupported preprocess method : {0}".format(args.output_unit))

            if convert_flag:
                converted_info[audio_num] = new_sentence

            ## 넣기
            fileinfo.append("{}\t{}".format(os.path.relpath(file_path, args.root), frames))
            durations.append(frames)
            audio_nums.append(audio_num)
            raw_sentences.append(raw_sentence)
            new_sentences.append(new_sentence)
    print("총 무시된 숫자 : ", limit_count+remove_count)
    print("길이를 넘겨서 무시된 숫자 : ", limit_count)
    print("숫자등이 있어서 무시된 숫자 : ", remove_count)

    return fileinfo, durations, texts, audio_nums, transcripts, raw_sentences, new_sentences, converted_info, additional_texts, additional_transcripts



def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    args.root = os.path.realpath(args.root)

    ## --dataset_path 에 있어야 하는 폴더들
    #for folder in ['KsponSpeech_01','KsponSpeech_02','KsponSpeech_03','KsponSpeech_04','KsponSpeech_05','KsponSpeech_eval']:
    #    if folder not in os.listdir(args.root):
    #        assert os.path.isdir(folder), "root 위치에 해당 폴더가 반드시 필요합니다. [{}]".format(folder)

    assert os.path.isdir(args.script_path), "aihub에서 제공해주는 스크립트 폴더를 넣어주시기 바랍니다. script_path : [{}]".format(args.script_path)

    ## Info 파일 불러오기
    info_data = {}
    if args.do_info:
        ## info 파일 불러오기
        info_data = load_info(args.info)

    ## .trn 확장자만 확인함
    file_list = [file for file in os.listdir(args.script_path) if Path(file).suffix == '.trn']
    assert len(file_list) > 0, "스크립트 파일이 한개도 없네요 [{}]".format(args.script_path)

    ## 스크립트 읽어오기.
    script_name = 'train.trn'
    if script_name in file_list:
        print("generate [{}]".format(script_name))
        fileinfo, durations, texts, audio_nums, transcripts, raw_sentences, new_sentences, converted_info,  additional_texts, additional_transcripts = load_script(args, os.path.join(args.script_path, script_name), info_data, token_limit=args.token_limit)
        fileinfo = np.array(fileinfo)
        durations = np.array(durations)
        texts = np.array(texts)
        transcripts = np.array(transcripts)

        ## 추가용
        additional_texts = np.array(additional_texts)
        additional_transcripts = np.array(additional_transcripts)

        ## lexicon 만들기
        save_lexicon(args, texts, lexicon_name='lexicon.lst')
        ## dictionary 저장
        save_dict(args, transcripts, dict_name='dict.ltr.txt', alphabet_name='alphabet.txt')

        ## 추가용 만들기
        if args.additional_output_unit is not None:
            ## lexicon 만들기
            save_lexicon(args, additional_texts, lexicon_name='add_lexicon.lst')
            ## dictionary 저장
            save_dict(args, additional_transcripts, dict_name='add_dict.ltr.txt', alphabet_name='add_alphabet.txt')

        #save_wrong_script(args, 'train_wrong',transcripts, fileinfo, raw_sentences, new_sentences)
        save_converted_info(args, 'train_converted', converted_info)

        ## train 이랑 dev 나눠서 저장
        train_ids = [idx for idx, num in enumerate(audio_nums)]
        limit_idx = len(train_ids)
        if args.time is not None:
            random.shuffle(train_ids)
            assert args.time in ['10min', '1hour', '10hour', '100hour'], '설정 재대로 해라...'
            time_limit = 0
            if args.time == '10min':
                ## 16000 hz * 60초 * 10분
                time_limit = 16000 * 60 * 10
            if args.time == '1hour':
                ## 16000 hz * 60초 * 60분 * 1
                time_limit = 16000 * 60 * 60 * 1
            if args.time == '10hour':
                ## 16000 hz * 60초 * 60분 * 10
                time_limit = 16000 * 60 * 60 * 10
            if args.time == '100hour':
                ## 16000 hz * 60초 * 60분 * 100
                time_limit = 16000 * 60 * 60 * 100

            limit_idx = find_index(durations[train_ids], time_limit)

        save_files(args, 'train', args.root, fileinfo[train_ids[:limit_idx]], texts[train_ids[:limit_idx]],
                   transcripts[train_ids[:limit_idx]])
        ## 추가용 만들기
        if args.additional_output_unit is not None:
            save_files(args, 'add_train', args.root, fileinfo[train_ids[:limit_idx]], additional_texts[train_ids[:limit_idx]],
                       additional_transcripts[train_ids[:limit_idx]])

    ## 스크립트 읽어오기.
    script_name = 'dev.trn'
    if script_name in file_list:
        print("generate [{}]".format(script_name))
        fileinfo, durations, texts, audio_nums, transcripts, raw_sentences, new_sentences, converted_info, additional_texts, additional_transcripts = load_script(args, os.path.join(args.script_path, script_name), info_data)
        save_files(args, 'dev', args.root, fileinfo, texts, transcripts)

        ## 추가용 만들기
        if args.additional_output_unit is not None:
            save_files(args, 'add_dev', args.root, fileinfo, additional_texts, additional_transcripts)

        #save_wrong_script(args, 'dev_wrong', transcripts, fileinfo, raw_sentences, new_sentences)
        save_converted_info(args, 'dev_converted', converted_info)

    ## 스크립트 읽어오기.
    script_name = 'eval_other.trn'
    if script_name in file_list:
        print("generate [{}]".format(script_name))
        fileinfo, durations, texts, audio_nums, transcripts, raw_sentences, new_sentences, converted_info, additional_texts, additional_transcripts = load_script(args, os.path.join(args.script_path,
                                                                                             script_name), info_data)
        save_files(args, 'eval_other', args.root, fileinfo, texts, transcripts)

        ## 추가용 만들기
        if args.additional_output_unit is not None:
            save_files(args, 'add_eval_other', args.root, fileinfo, additional_texts, additional_transcripts)

        #save_wrong_script(args, 'eval_other_wrong', transcripts, fileinfo, raw_sentences, new_sentences)
        save_converted_info(args, 'eval_other_converted', converted_info)

    ## 스크립트 읽어오기.
    script_name = 'eval_clean.trn'
    if script_name in file_list:
        print("generate [{}]".format(script_name))
        fileinfo, durations, texts, audio_nums, transcripts, raw_sentences, new_sentences, converted_info, additional_texts, additional_transcripts = load_script(args, os.path.join(args.script_path,
                                                                                             script_name), info_data)
        save_files(args, 'eval_clean', args.root, fileinfo, texts, transcripts)

        ## 추가용 만들기
        if args.additional_output_unit is not None:
            save_files(args, 'add_eval_clean', args.root, fileinfo, additional_texts, additional_transcripts)
        #save_wrong_script(args, 'eval_clean_wrong', transcripts, fileinfo, raw_sentences, new_sentences)
        save_converted_info(args, 'eval_clean_converted', converted_info)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()


    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    _print_config(args)

    main(args)
