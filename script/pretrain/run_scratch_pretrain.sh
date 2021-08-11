# RUN Scratch pretrain
# task.data : path of preprocessed manifest folder
# checkpoint.save_dir : path to save pretrain checkpoint
# task.del_silence : whether use silence options which indicate removing prolonged silence in voice

## before run code, please check config files to modify options required.

python -W ignore fairseq_cli/hydra_train.py \
  task.data=$(realpath .)/transcriptions/ksponspeech/character_spelling \
  checkpoint.save_dir=$(realpath .)/save_checkpoint/pretrain/scratch_pretrain \
  task.del_silence=True \
  --config-dir configs/pretrain \
  --config-name scratch_pretrain


