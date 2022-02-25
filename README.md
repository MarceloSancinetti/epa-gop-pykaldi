# Pronunciation scoring on Epa-DB using Pykaldi

This repository has the complete code and examples to obtain the results in the paper 'A transfer learning based approach for pronunciation scoring' by Sancinetti, Vidal Bonomi and Ferrer, accepted at ICASSP 2022. It is meant to facilitate experimentation with Epa-DB, a database of non-native English speech by Spanish speakers from Argentina intended for research on mispronunciation detection and development of pronunciation scoring systems. 

Two pronunciation scoring systems are available: (1) a standard goodness of pronunciation (GOP) DNN algorithm and (2) a GOP-FT algorithm, where we replace the last layer of an ASR DNN, and train the resulting model for the pronunciation scoring task. Both systems use a PyTorch acoustic model based on Kaldi's TDNN-F acoustic model so a script is provided to convert Kaldi's model to PyTorch.


## Table of Contents
* [Prerequisites](#prerequisites)
* [How to install](#how-to-install)
* [How to run](#how-to-run)
* [Data preparation](#data-preparation)
* [How to run the GOP recipe](#how-to-run-the-GOP-recipe)
* [How to run the GOP-FT recipe](#how-to-run-the-GOP-FT-recipe)
* [Copyright](#copyright)
* [References](#references)

## Prerequisites
1. [Kaldi](http://kaldi-asr.org/) installed.
2. TextGrid managing library installed using pip. Instructions at this [link](https://pypi.org/project/praat-textgrids/).
3. The EpaDB database downloaded (you can ask for it at jvidal@dc.uba.ar). 

## How to install
To install this repository, follow this steps:

1. Clone this repository:
```
git clone https://github.com/MarceloSancinetti/epa-gop-pykaldi.git
```
2. Install the requirements:
```
pip install -r requirements.txt
```
3. Install PyKaldi:
Follow instructions from https://github.com/pykaldi/pykaldi#installation

## How to run
To run the GOP-DNN system go to How to run the GOP recipe. 
To replicate the experiments in the paper go to How to run the GOP-FT recipe. 
Both recipes assume a previous common step of data preparation. 

## Data preparation
Before using any of the systems, it is necessary to run the data preparation script. This step handles feature extraction, downloads the Librispeech ASR acoustic model from OpenSLR, converts said model to PyTorch and creates forced alignments and training labels. This should only be done once unless EpaDB is updated, in which case new features, labels, and alignments need to be generated.

To run data preparation use:
```
python run_dataprep.py --config configs/dataprep.yaml
```

## How to run the GOP recipe
For computing GOP [1], we recreate the official Kaldi [2] recipe in PyKaldi [3]. We use the Kaldi Librispeech ASR model, a TDNN-F acoustic model, ported to PyTorch in the previous stage. 

To run the GOP recipe use:
```
python run_gop.py --config configs/gop.yaml --from STAGE --to STAGE
```
STAGE can be one of the following: prep, gop, evaluate.

To run all stages, use ``` --from prep --to evaluate ```

## How to run the GOP-FT recipe
In the GOP-FT, we explore the use of a simple transfer learning-based approach for pronunciation scoring. As in [4], we replace the last layer of an ASR DNN, and train the resulting model for the pronunciation scoring task.

We explore two approaches for fine-tuning the new model: (1) **LayO**, where only the new output layer is trained, keeping all other parameters frozen at their pre-trained values, and (2) **LayO+1**, where the last hidden layer is also trained. In the second case, we train the model in two stages: first, only the output layer is trained over several epochs and then the second to last layer is unfrozen and both layers are further fine-tuned.

The config directory includes two configuration files that list the hyperparameters used during training for both cases. 
To run the **LayO** experiment, use: 
```
python run_experiment.py --config configs/lay1_lr01_bs32_bnlst_do40_npc_wthr50_steplr10g9.yaml --from STAGE --to STAGE --device DEVICE
```
STAGE can be one of the following: prep, train, scores, evaluate.
DEVICE can be either cpu of cuda.
Add the --heldout option to run on heldout set.

To run the **LayO+1** experiment, use:
```
python run_experiment.py --config configs/lay2_lr01_bs32_bnlst_do40_npc_wthr50_steplr10g9_epoch300__lr001_steplr10g9.yaml  --from STAGE --to STAGE --device DEVICE
```
The examples correspond to the hyperparameters resulting in the system with the best performance. Custom configuration files can be created if you wish to tweak the hyperparameter values by following the examples at the config directory. 


## Copyright
The code in this repository and the EpaDB database were developed at the Speech Lab at Universidad de Buenos Aires, Argentina and are freely available for research purposes. 

If you use the EpaDB database, please cite the following paper:

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

If you use the code in this repository, please cite the following paper:

*M. Sancinetti, J. Vidal, C. Bonomi, L.Ferrer, "A transfer learning approach for pronunciation scoring", [Arxiv](https://arxiv.org/pdf/2111.00976.pdf)*

```
@article{sancinetti2021transfer,
  title={A transfer learning based approach for pronunciation scoring},
  author={Sancinetti, Marcelo and Vidal, Jazmin and Bonomi, Cyntia and Ferrer, Luciana},
  journal={arXiv preprint arXiv:2111.00976},
  year={2021}
}
}
```

## References
* [1] S.M. Witt and S.J. Young, “Phone-level pronunciation scoring and assessment for interactive language learning,” Speech
communication, vol. 30, no. 2-3, pp. 95–108, 2000.
* [2] D. Povey, A. Ghoshal, G. Boulianne, L. Burget, O. Glembek, N. Goel, M. Hannemann, P. Motlicek, Y. Qian, P. Schwarz, et al., “The kaldi speech     recognition toolkit,” in IEEE 2011 workshop on automatic speech recognition and understanding. IEEE Signal Processing Society, 2011, number CONF.
* [3] C. Dogan, Martinez V., Papadopoulos P., and Narayanan S.S, “Pykaldi: A python wrapper for kaldi,” in Acoustics, Speech and Signal Processing (ICASSP), 2018 IEEE International
Conference on. IEEE, 2018.
* [4] H. Huang, H. Xu, Y. Hu, and G. Zhou, “A transfer learning approach to goodness of pronunciation based automatic mispronunciation detection,” The Journal of the Acoustical Society of
America, vol. 142, no. 5, pp. 3165–3177, 2017






