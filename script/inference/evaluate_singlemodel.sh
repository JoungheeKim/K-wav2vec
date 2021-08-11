# Evaluate fine-tuned model

## select MANIFEST PATH used in fine-tuning
MANIFEST_PATH=$(realpath .)/transcriptions/ksponspeech/character_spelling

## fine-tuned checkpoint
CHECKPOINT_PATH=$(realpath .)/save_checkpoint/finetune/ksponspeech/single_model_transformer/checkpoint_best.pt  ## it is dummy, please modify it

## single model use 'audio_pretraining', multi model use 'audio_multitraining' for task
TASK=audio_pretraining

## we only support 'beam'
DECODER=beam

## length of beam, default:100
BEAM=100

## SUBSET indiates evaluation set. our manifest include dev, eval_clean, eval_other
## Therefore, modele is evaluated in different subsets.
for SUBSET in dev eval_clean eval_other; do

    ## We report our results with CSV file.
    ## Put csv path and name in EXPERIMENT_DIR
    EXPERIMENT_DIR=experiments/ksponspeech/singlemodel_${SUBSET}.csv

    ## Put path to save log during evaluation.
    RESULTS_PATH=eval_log/ksponspeech/singlemodel_${SUBSET}

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
       --experiments-dir ${EXPERIMENT_DIR}
done