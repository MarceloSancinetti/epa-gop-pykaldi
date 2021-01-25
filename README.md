# PyKaldi GOP-DNN on Epa-DB

This repository has the tools to run a PyKaldi GOP-DNN algorithm on Epa-DB, a database of non-native English speech by Spanish speakers from Argentina. It uses a PyTorch acoustic model based on Kaldi's TDNN-F acoustic model. A script is provided to convert Kaldi's model to PyTorch. Kaldi's model must be downloaded separately from the Kaldi website

If you use this code or the Epa database, please cite the following paper:

*J. Vidal, L. Ferrer, L. Brambilla, "EpaDB: a database for the development of pronunciation assessment systems", [isca-speech](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/1839.html)*

```
@article{vidal2019epadb,
  title={EpaDB: a database for development of pronunciation assessment systems},
  author={Vidal, Jazmin and Ferrer, Luciana and Brambilla, Leonardo},
  journal={Proc. Interspeech 2019},
  pages={589--593},
  year={2019}
}
```


## Table of Contents
* [Introduction](#introduction)
* [Prerequisites](#prerequisites)
* [How to install](#how-to-install)
* [How to run](#how-to-run)
* [Notes on Kaldi-DNN-GOP](#Notes-on-Kaldi-DNN-GOP)
* [References](#references)


## Introduction

This toolkit is meant to facilitate experimentation with Epa-DB by allowing users to run a state-of-the-art baseline system on it.
Epa-DB, is a database of non-native English speech by argentinian speakers of Spanish. It is intended for research on mispronunciation detection
and development of pronunciation assessment systems.
The database includes recordings from 30 non-native speakers of English, 15 male and 15 female, whose first language (L1) is Spanish from Argentina (mainly of the Rio de la Plata dialect).
Each speaker recorded 64 short English phrases phonetically balanced and specifically designed to globally contain all the sounds difficult to pronounce for the target population.
All recordings were annotated at phone level by expert raters.

For more information on the database, please refer to the [documentation](https://drive.google.com/file/d/1jEvqeAXTLKRAYJXTQAvfsc3Qye6vOb5o/view?usp=sharing) or [publication](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/1839.html)

If you are only looking for the EpaDB corpus, you can download it from this [link](https://drive.google.com/file/d/12wD6CzVagrwZQcMTgTxw2_7evjZmPQym/view?usp=sharing).


## Prerequisites

1. [Kaldi](http://kaldi-asr.org/) installed.

2. TextGrid managing library installed using pip. Instructions at this [link](https://pypi.org/project/praat-textgrids/).

3. [The EpaDB database](https://drive.google.com/file/d/1jEvqeAXTLKRAYJXTQAvfsc3Qye6vOb5o/view?usp=sharing) downloaded. Alternative [link](https://www.dropbox.com/s/m931q0vch1qhzzx/epadb.zip?dl=0).

4. [Librispeech ASR model](https://kaldi-asr.org/models/m13)


## How to install

To install this repository, do the following steps:

1. Clone this repository:
```
git clone https://github.com/MarceloSancinetti/epa-gop-pykaldi.git
```

2. Download Librispeech ASR acoustic model from Kaldi and move it or link it inside the top directory of the repository:

```
wget https://kaldi-asr.org/models/13/0013_librispeech_v1_chain.tar.gz
tar -zxvf 0013_librispeech_v1_chain.tar.gz
```

3. Convert the acoustic model to text format:

```
nnet3-copy --binary=false exp/chain_cleaned/tdnn_1d_sp/final.mdl exp/chain_cleaned/tdnn_1d_sp/final.txt
```

4. Install the requirements:

```
pip install -r requirements.txt
```

5. Install PyKaldi:

Follow instructions from https://github.com/pykaldi/pykaldi#installation

6. Convert the acoustic model to Pytorch:

```
python convert_chain_to_pytorch.py
```
