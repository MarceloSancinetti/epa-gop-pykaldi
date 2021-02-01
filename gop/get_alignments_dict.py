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
import pickle
from random import shuffle

def get_phone_from_transition_id(id, df_phones_pure):
    return df_phones_pure.loc[df_phones_pure['transition_id'] == str(id),'phone_pure'].values[0]

def removeSymbols(str, symbols):
    for symbol in symbols:
        str = str.replace(symbol,'')
    return str

def get_alignments(path_alignements, alignments_dict, df_phones_pure):
    for l in open(path_alignements+"align_output", encoding="utf8", errors='ignore').readlines():
        l=l.split()
        #Get transitions alignments
        if len(l) > 3 and l[1] == 'transitions':
            waveform_name = l[0]
            alignment_array = []
            current_phone_transition = int(removeSymbols(l[2],['[',']',',']))
            current_phone = get_phone_from_transition_id(current_phone_transition, df_phones_pure)
            transitions = []
            #alignments_dict[waveform_name] = {}
            for i in range(2, len(l)):
                transition_id = int(removeSymbols(l[i],['[',']',',']))
                phone = get_phone_from_transition_id(transition_id, df_phones_pure)
                if phone != current_phone:
                    alignment_array.append(transitions)
                    transitions = []
                    current_phone_transition = transition_id
                    current_phone = get_phone_from_transition_id(current_phone_transition, df_phones_pure)
                transitions.append(transition_id)
            alignments_dict[waveform_name]['transitions'] = alignment_array

        #Get phones alignments
        if len(l) > 3 and l[1] == 'phones':
            waveform_name = l[0]
            phones = []
            alignments_dict[waveform_name] = {}
            for i in range(2, len(l),3):
                current_phone = removeSymbols(l[i],['[',']',',',')','(','\''])
                phones.append(current_phone)
            alignments_dict[waveform_name]['phones'] = phones
    return alignments_dict





def main(data_path="."):
    df_phones_pure = pd.read_pickle('phones_pure_epa.pickle')

    #path_show_alignments = "alignments/"
    path_show_alignments = ""

    alignments_dict = {}

    get_alignments(path_show_alignments, alignments_dict, df_phones_pure)

    with open('alignments.pickle', 'wb') as handle:
        pickle.dump(alignments_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
        main()
