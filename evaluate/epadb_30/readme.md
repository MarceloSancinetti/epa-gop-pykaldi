About EpaDB

EpaDB is a speech database intended for research in pronunciation scoring. The corpus includes audios from 50 Spanish speakers (25 males and 25 female) from Argentina reading phrases in English. Each speaker recorded 64 short phrases containing sounds hard to pronounce for this population adding up to a total of 20 hours of speech.

In addition to the recordings, we manually annotated each phrase at phonetic level by two expert annotators from Argentina. One of the annotations is complete, the other covers only four phrases per speaker. We intend to complete the process and also add a third English native annotator in the course of the year.

For more information on the annotations, please refer to the paper in this folder.

The database is an effort of the Speech Lab at the Laboratorio de Inteligencia Artificial Aplicada from the Universidad de Buenos Aires and was partially funded by Google by a Google Latin America Reseach Award in 2018.


Overview

In the downloaded zip file you will encounter the following folders organized by speaker:

- waveforms: contains all the speech recordings.
- transcriptions: contains the prompts in .lab format.
- annotations_1: contains the set of complete manual annotations in TextGrid format.
- annotations_2: contains a subset of manual annotations in TextGrid format by a second annotator.
- reference_transcriptions: the file with all the possible reference transcriptions we considered.


Note that some of the recordings may be missing due to low quality or recording problems.
Manual annotation was performed at phone level using ARPA-bet and an ARPA-bet like extension to account for allophonic variations and Spanish sounds. For more informations refer to the paper in this folder.
Speakers 89 and 18 do not contain the set of second annotations.

Note on the TextGrid files

Each TextGrid file contains 4 tiers:

- Words: manually corrected word level alignments.
- Transcription: manually corrected phoneme level alignments containing one of the possible correct transcriptions.
- Annotation: phone level annotations by the expert annotators.
- Score: an overall score provided by the annotator in a scale from 1 to 5.


If you want to replicate our results in pronunciation scoring using Kaldi-GOP or find a code to compute labels from the manual annotations you can run the code provided in our github repository: https://github.com/JazminVidal/gop-dnn-epadb


If you are interested in using EpaDB in your publications, please cite the following paper:
@article{vidal2019epadb,  title={EpaDB: a database for development of pronunciation assessment systems},  author={Vidal, Jazmin and Ferrer, Luciana and Brambilla, Leonardo},  journal={Proc. Interspeech 2019},  pages={589--593},  year={2019}}
