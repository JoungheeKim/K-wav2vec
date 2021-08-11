import argparse
import os
import numpy as np
import re
import librosa
import json
from sklearn.model_selection import train_test_split
from pathlib import Path
import soundfile as sf



def load_json(temp_path):
    wav_list = list()
    text_list = list()
    with open(temp_path) as f:
        data_list=json.load(f)
        for temp_data in data_list:
            wav = temp_data['wav']
            text = temp_data['text']
            wav_list.append(wav)
            text_list.append(text)
    return wav_list, text_list

## Fairseq 스타일로 변환하기
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default='/code/gitRepo/data/clovacall', metavar="DIR",
        help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--dest", default='/code/gitRepo/data/clovacall/script', type=str, metavar="DIR",
        help="output directory"
    )
    parser.add_argument(
        "--dev_portion", default=600, type=int,
        help="dev portion"
    )
    return parser


def load_audio(file_path):
    ext = Path(file_path).suffix

    if ext in ['.wav', '.flac']:
        wav, sr = librosa.load(file_path, sr=16000)
    elif ext == '.pcm':
        wav = np.memmap(file_path, dtype='h', mode='r').astype('float32') / 32767
        sr = 16000
    elif ext in ['.raw', '.RAW']:
        wav, sr = sf.read(file_path, channels=1, samplerate=16000,
                          format='RAW', subtype='PCM_16')
    else:
        raise ValueError("Unsupported preprocess method : {0}".format(Path(file_path).suffix))

    return wav, sr



def make_script(args, dir_path, wav_list, text_list):
    fileinfo = list()
    except_files = list()

    ## 숫자, 영어, 한글을 제외하고 특수문자 전체 제거(.?/)
    pattern = '[^\w\s]'

    for audio_path, raw_sentence in zip(wav_list, text_list):
        audio_path = os.path.join(dir_path, audio_path)
        audio_path = os.path.realpath(audio_path)

        try:
            wav, sr = load_audio(audio_path)

            ## 길이
            new_sentence = re.sub(pattern=pattern, repl='', string=raw_sentence).strip()
            fileinfo.append("{} :: {}".format(os.path.relpath(audio_path, args.root), new_sentence))

        except:
            except_files.append(audio_path)

    return fileinfo, except_files

def save_trn(args, fileinfo, file_name='train'):

    print("save files [{}]".format(file_name))
    with open(os.path.join(args.dest, "{}.trn".format(file_name)), 'w', encoding='UTF8') as trn_out:
        for trn_item in fileinfo:
            print(trn_item, file=trn_out)

def main(args):
    ## 데이터 형태가 반드시 일치해야 합니다.
    """
    |-clovacall
         |------train_ClovaCall.json
         |------test_ClovaCall.json
         |------wavs_train
                    |----------41_0514_688_0_07118_05.wav
                    |----------41_0509_714_0_08568_02.wav
         |------wavs_test
                    |----------41_0514_301_0_07111_09.wav
                    |----------41_0515_577_0_04088_07.wav
    """

    train_json_name = 'train_ClovaCall.json'
    test_json_name = 'test_ClovaCall.json'
    train_folder_name = 'wavs_train'
    test_folder_name = 'wavs_test'

    assert os.path.isdir(args.root), "폴더가 없습니다. 다시 한번 확인해 주세요 [{}]".format(args.root)
    folder_list = os.listdir(args.root)
    for item in [train_json_name, test_json_name, train_folder_name, test_folder_name]:
        assert item in folder_list, "파일이 없습니다. 다시 한번 확인해 주세요. [{}]".format(item)

    os.makedirs(args.dest, exist_ok=True)

    train_wav, train_text = load_json(os.path.join(args.root, train_json_name))
    train_wav, valid_wav, train_text, valid_text = train_test_split(train_wav, train_text, test_size=args.dev_portion)
    test_wav, test_text = load_json(os.path.join(args.root, test_json_name))


    train_folder= os.path.join(args.root, train_folder_name)
    fileinfo, except_files = make_script(args, train_folder, train_wav, train_text)
    save_trn(args, fileinfo, 'train')
    print("저장된 파일", len(fileinfo))
    print("제외된 파일", len(except_files))

    dev_folder = os.path.join(args.root, train_folder_name)
    fileinfo, except_files = make_script(args, dev_folder, valid_wav, valid_text)
    save_trn(args, fileinfo, 'dev')
    print("저장된 파일", len(fileinfo))
    print("제외된 파일", len(except_files))

    test_folder = os.path.join(args.root, test_folder_name)
    fileinfo, except_files = make_script(args, test_folder, test_wav, test_text)
    save_trn(args, fileinfo, 'eval_clean')
    print("저장된 파일", len(fileinfo))
    print("제외된 파일", len(except_files))

    return


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()


    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    _print_config(args)

    main(args)
