# Evaluate fine-tuned model

## select MANIFEST PATH used in fine-tuning(for multi model, need to use manifest supporting addtional script)
MANIFEST_PATH=$(realpath .)/transcriptions/ksponspeech/grapheme_character_spelling

## fine-tuned checkpoint
CHECKPOINT_PATH=$(realpath .)/save_checkpoint/finetune/ksponspeech/multi_model/checkpoint_best.pt  ## it is dummy, please modify it

## single model use 'audio_pretraining', multi model use 'audio_multitraining' for task
TASK=audio_multitraining

## we only support 'beam'
DECODER=beam

## length of beam, default:100
BEAM=100

## joint decoding has contirbution weight to balance grapheme and syllable
## put between 0.0~1.0
CONTRIBUTION_WEIGHT=0.5

## SUBSET indiates evaluation set. our manifest include dev, eval_clean, eval_other
## Therefore, modele is evaluated in different subsets.
for SUBSET in dev; do

    ## We report our results with CSV file.
    ## Put csv path and name in EXPERIMENT_DIR
    EXPERIMENT_DIR=experiments/ksponspeech/multi_model_${SUBSET}.csv

    ## Put path to save log during evaluation.
    RESULTS_PATH=eval_log/ksponspeech/multi_model_${SUBSET}

    python inference/beam_search.py ${MANIFEST_PATH} \
       --task ${TASK} \
       --checkpoint-path ${CHECKPOINT_PATH} \
       --gen-subset ${SUBSET} \
       --results-path ${RESULTS_PATH} \
       --decoder ${DECODER} \
       --criterion multi_ctc \
       --labels ltr \
       --post-process letter \
       --beam=${BEAM} \
       --add-weight=${CONTRIBUTION_WEIGHT} \
       --experiments-dir ${EXPERIMENT_DIR}
done