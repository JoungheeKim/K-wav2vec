# RUN Further pretrain
# task.data : path of preprocessed manifest folder
# checkpoint.save_dir : path to save pretrain checkpoint
# task.del_silence : whether use silence options which indicate removing prolonged silence in voice
# checkpoint.finetune_from_model : checkpoints which is used for init state. You can download english checkpoints in Fairseq github "https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md"

## before run code, please check config files to modify options required.

python -W ignore fairseq_cli/hydra_train.py \
    task.data=$(realpath .)/transcriptions/ksponspeech/character_spelling \
    checkpoint.save_dir=$(realpath .)/save_checkpoint/pretrain/further_pretrain \
    task.del_silence=True \
    checkpoint.finetune_from_model=$(realpath .)/save_checkpoint/pretrain/english_pretrain/wav2vec_small.pt \
    --config-dir configs/pretrain \
    --config-name further_pretrain