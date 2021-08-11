# Preprocess for Ksponspeech dataset
# Paper : https://www.mdpi.com/2076-3417/10/19/6936
# Dataurl : https://aihub.or.kr/aidata/105

# All related data must in same directory described below
# RootDir
#   |-------KsponSpeech_01
#   |-------KsponSpeech_02
#   |-------KsponSpeech_03
#   |-------KsponSpeech_04
#   |-------KsponSpeech_05
#   |-------KsponSpeech_eval

# Put your absolute 'RootDir' path below
ROOT_DIR=/code/gitRepo/data/aihub/ksponspeech ## it is dummy, please modify it

# Put your Script path which is also given with Ksponspeech dataset
SCRIPT_PATH=/code/gitRepo/data/aihub/ksponspeech/KsponSpeech_scripts ## it is dummy, please modify it


# Select for preprocess output
# You can choose either 'grapheme' or 'character'
# Here, character means syllable block which is korean basic character
OUTPUT_UNIT=character

# if length of script is over limit, it is exculded as described in https://arxiv.org/abs/2009.03092
LIMIT=200

# Ksponspeech data support dual transcriptions including phonetic and orthographic.
# Therefore, select transcription type [phonetic, spelling]
# Here, phonetic : phonetic, orthographic : spelling
#PROCESS_MODE=phonetic
PROCESS_MODE=spelling

# Put your absolute destination path
DESTINATION=$(realpath .)/transcriptions/ksponspeech

## Run preprocess code
python preprocess/make_manifest.py \
     --root ${ROOT_DIR} \
     --output_unit ${OUTPUT_UNIT} \
     --do_remove \
     --preprocess_mode ${PROCESS_MODE} \
     --token_limit ${LIMIT} \
     --dest ${DESTINATION}/${OUTPUT_UNIT}_${PROCESS_MODE} \
     --script_path ${SCRIPT_PATH}

