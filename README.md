# K-Wav2vec 2.0

This official implementation of ["K-Wav2vec 2.0: Automatic Speech Recognition based on Joint Decoding of Graphemes and Syllables"](https://arxiv.org/abs/2110.05172)

## Requirements and Installation

 - PyTorch version >= 1.7.1
 - Python version >= 3.6
 - To install K-wav2vec and develop locally:
```bash
git clone https://github.com/JoungheeKim/K-wav2vec.git
cd K-wav2vec

## install essential library
pip install -r requirements.txt

## install locally
python setup.py develop
```
 - We only test this implementation in Ubuntu 18.04.
 - DockerFile is also supported in this repo.


## Instructions
 - We support script examples to execute code easily(check `script` folder)
 - Following this instruction give you exact matched results.
```bash
# Guilde to make multi-model with Ksponspeech(orthographic transcription) 

# [1] preprocess dataset & make manifest
bash script/preprocess/make_ksponspeech_script_for_mulitmodel.sh

# [2] further pre-train the model
bash script/pretrain/run_further_pretrain.sh
 
# [3] fine-tune the model
bash script/finetune/run_ksponspeech_multimodel.sh

# [4] inference the model
bash script/inference/evaluate_multimodel.sh
```


## Pretrained model
 - E-Wav2vec 2.0 : Wav2vec 2.0 pretrained on Englsih dataset released by Fairseq(-py)
 - K-Wav2vec 2.0 : The model further pretrained on Ksponspeech by using Englsih model
    - Fairseq Version : If you want to fine-tune your model with fairseq framework, you can download with this [LINK](https://drive.google.com/file/d/12EeFehYzsec-axu0E7P8uCyyXkbTYLJS/view?usp=sharing)
    - Huggingface Version : If you want to fine-tune your model with huggingface framework, you can download with this [LINK](https://drive.google.com/file/d/1a_BHLFPRb8OHKnwL4r3tiRZMw9R8k2B4/view?usp=sharing)


## Dataset
 - [Ksponspeech](https://aihub.or.kr/aidata/105) : Open-domain dialog corpus
 - [Clovacall](https://github.com/clovaai/ClovaCall) : Call-based speech corpus for reservation


## Acknowledgments
 - Our code was modified from [fairseq](https://github.com/pytorch/fairseq) codebase. We use the same license as fairseq.
 - The preprocessing code was developed with reference to [Kospeech](https://github.com/sooftware/KoSpeech).

## License
Our implementation code(-py) is MIT-licensed. The license applies to the pre-trained models as well.


 
