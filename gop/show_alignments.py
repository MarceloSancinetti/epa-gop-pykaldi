 #!usr/bin/env python3 

from IPython import embed
from pathlib import Path 
import numpy as np 
import pandas as pd 
import tqdm 
import glob 
import librosa 
import joblib
import math
import os
import os.path
import shutil
from os import listdir
from os.path import isfile, isdir
import argparse
import math
import os, errno, re
import os.path as path
from random import shuffle


def show_alignments(path_output_MFA, path_output, path_output_show_alignments, path_kaldi):
	print("Generar la lista de lista de transiciones dado los alineamientos")

	if os.path.isdir(path_output_show_alignments):
		shutil.rmtree(path_output_show_alignments)


	if not os.path.isdir(path_output_show_alignments):
		os.mkdir(path_output_show_alignments)


	files=listdir(path_output)	
	for spkr in tqdm.tqdm(files):
		os.system(path_kaldi+"show-alignments "+ path_output_MFA+spkr+"/dictionary/phones.txt "+path_output_MFA+spkr+"/align/final.mdl ark:"+path_output_MFA+spkr+"/align/ali.0  > "+path_output_show_alignments+spkr+".txt")  



def get_alignments(path_list_waveforms, path_alignements, alignments_dict):
    output_ = []
    f = open(path_list_waveforms, "r")
    for line in tqdm.tqdm(f):
        spkr = line.replace('\n', '').split('-')[0]
        for l in open(path_alignements+spkr+".txt", encoding="utf8", errors='ignore').readlines():

            l=l.split()
            if len(l) > 2:
                waveform_name = l[0]
                alignment_array = []
                annotation_array = []
                senones_array = []
                if l[1] == '[':
                    for i in range(1, len(l)):
                        if l[i] == '[':
                            l_aux = []
                        if l[i] != '[' and l[i] != ']':
                            l_aux.append(l[i])
                        if l[i] == ']':
                            alignment_array.append(l_aux)

                    alignments_dict[waveform_name] = alignment_array
    return alignments_dict





def main(data_path="."):
    path_show_alignments = "alignments/"

    alignments_dict = {}

    phone_dict = get_phones(path_phone)

    path_list_waveforms= "validate_360hs.list"
    get_alignments(path_list_waveforms, path_show_alignments, alignments_dict)

if __name__ == "__main__":
        main()

                                                                                                                                                                                          198,1-8     Final
