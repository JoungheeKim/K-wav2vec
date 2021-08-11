# RUN finetune with Multi-task architecture
# task.data : path of preprocessed manifest folder (multi-model must run in script which has grapheme and character vocabulary)
# checkpoint.save_dir : path to save finetune checkpoint
# task.del_silence : whether use silence options which indicate removing prolonged silence in voice
# model.additional_layers : number of transformer layer for syllable encoder which contains stacks of transformers and projection layers : default=2
# checkpoint.best_checkpoint_metric : evaluation metric for development set. In multi-model, 'wer' means evaluate model with grapheme outputs, and 'add_wer' means evaluate model with syllabel outputs.
# model.w2v_path : pre-trained checkpoints to use for fine-tuining (use either further-pretrained model or scratch-pretrained model)

## before run code, please check config files to modify options required.

python -W ignore fairseq_cli/hydra_train.py \
  task.data=$(realpath .)/transcriptions/ksponspeech/grapheme_character_spelling \
  checkpoint.save_dir=$(realpath .)/save_checkpoint/finetune/ksponspeech/multi_model \
  task.del_silence=True \
  model.additional_layers=2 \
  checkpoint.best_checkpoint_metric=add_wer \
  model.w2v_path=$(realpath .)/save_checkpoint/pretrain/further_pretrain/checkpoint_best.pt \
  --config-dir configs/finetune/multi \
  --config-name 960h