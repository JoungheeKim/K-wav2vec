# Preprocess for Clovacall dataset
# Paper : https://arxiv.org/abs/2004.09367
# Dataurl : https://github.com/clovaai/ClovaCall


## 1. First make aihub style script with Clovacall Dataset
# All related data must in same directory described below
# RootDir
#   |-------wavs_train
#   |-------wavs_test

# Put your absolute 'RootDir' path below
ROOT_DIR=/code/gitRepo/data/clovacall ## it is dummy, please modify it

# Put your destination path to save your scripts
SCRIPT_PATH=$(realpath .)/transcriptions/clovacall/script

python preprocess/generate_script.py \
    --root=${ROOT_DIR} \
    --dest=${SCRIPT_PATH}





## 2. Second make manifest same as ksponspeech preprocess


# Select for preprocess output
# You can choose either 'grapheme' or 'character'
# Here, character means syllable block which is korean basic character
OUTPUT_UNIT=character

# Clovacall supports only orthographic transcriptions.
# Since we use same preprocess code of ksponspeech, preprocess options are carefully selected for Clovacall dataset
# Therefore, select orthographic transcription type [spelling]
PROCESS_MODE=spelling

# Put your absolute destination path
DESTINATION=$(realpath .)/transcriptions/clovacall

## Run preprocess code
python preprocess/make_manifest.py \
     --root ${ROOT_DIR} \
     --output_unit ${OUTPUT_UNIT} \
     --del_silence \
     --preprocess_mode ${PROCESS_MODE} \
     --dest ${DESTINATION}/${OUTPUT_UNIT}_${PROCESS_MODE} \
     --script_path ${SCRIPT_PATH}
