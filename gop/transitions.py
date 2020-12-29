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



#calcula el archivo de transiciones
#crea uno por speaker pero son todos iguales. 
def show_transitions(path_output_MFA, path_output, path_output_show_transitions, path_kaldi):
	print("Generar la lista de todas las posibles transiciones dado los alineamientos")

	if os.path.isdir(path_output_show_transitions):
		shutil.rmtree(path_output_show_transitions)

	if not os.path.isdir(path_output_show_transitions):
		os.mkdir(path_output_show_transitions)

	files=listdir(path_output)	
	for spkr in tqdm.tqdm(files):
		os.system(path_kaldi+"show-transitions "+ path_output_MFA+spkr+"/dictionary/phones.txt "+path_output_MFA+spkr+"/align/final.mdl "+path_output_MFA+spkr+"/align/final.occs > "+path_output_show_transitions+spkr+".txt")  





#toma las transiciones calculadas con la función show_transitions
def get_transitions(path_show_transitions, path_output_transitions):
    transitions_path = glob.glob(path_show_transitions+'*.txt')

    transitions_dict = {}

    for path_filename in tqdm.tqdm(transitions_path):

            f = open(path_filename, "r")
            phone = -1
            data = []
            for line in f:
                    line_array = line.split(' ')
                    if line_array[0] == 'Transition-state':
                            data = []
                            transition_state = line_array[1].split(":")[0]
                            phone = line_array[4]
                            hmm_state = line_array[7]
                            pdf = line_array[10]
                            pdf = pdf.split('\n')
                            pdf = pdf[0]

                            data.append(transition_state)
                            data.append(phone)
                            data.append(hmm_state)
                            data.append(pdf)
                    if line_array[1] == 'Transition-id':
                            transition_id = line_array[3]
                            transitions_dict[transition_id] = data + [transition_id]
                            
                    #if line_array[-1] == '[self-loop]\n':
                    #    data.append(transition_ids)
                    #    transitions_dict[transition_id] = data


    df_transitions = pd.DataFrame.from_dict(transitions_dict, orient='index', columns=['transition_state', 'phone_name', 'hmm_state', 'pdf', 'transition_id'])

    return df_transitions


def main(data_path="."):

    phones = open("phones.txt", "r")
    phones_dict = {}
    for line in tqdm.tqdm(phones):
            l = line.split()
            phones_dict[l[1]] = l[0]

    f = open("phones-pure.txt", "r")
    phones_pure_dict = {}
    for line in tqdm.tqdm(f):
            l = line.split()
            phones_pure_dict[l[1]] = l[0]

    output = []
    phone_to_pure_file = open("phone-to-pure-phone.int", "r")
    for line in tqdm.tqdm(phone_to_pure_file):
            l = line.split()
            output.append({'phone':l[0],
                            'phone_name':phones_dict[l[0]],
                            'phone_pure': l[1],
                            'phone_pure_name': phones_pure_dict[l[1]]})


    df_phone_pure = pd.DataFrame(output)

    path_show_transitions = "transitions/"
    path_output_transitions = "transitions.pickle"

    df_transitions = get_transitions(path_show_transitions, path_output_transitions)

    df = df_transitions.set_index('phone_name').join(df_phone_pure.set_index('phone_name'))
    df = df.reset_index().set_index("phone")

    df.to_pickle('phones_pure_epa.pickle')


if __name__ == "__main__":
    main()



