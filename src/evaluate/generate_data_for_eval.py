import os, errno, re
import os.path as path
from os import remove
import numpy as np
import textgrids # pip install git+https://github.com/Legisign/Praat-textgrids
from scipy.stats.stats import pearsonr
from IPython import embed
import pandas as pd
import joblib
import shutil
import argparse
import glob

from src.utils.reference_utils import *

def phonelist2str(phones):
    return " ".join(["%3s"%p for p in phones])

# Function that matches phone ints to phone symbols and loads them to a dictionary

def phones2dic(path):
    print(path)
    phones_dic = {}
    with open(path, "r") as fileHandler:
        line = fileHandler.readline()
        while line:
            print(line)
            l=line.split()
            phones_dic[int(l[1])] = l[0]
            line = fileHandler.readline()

    return phones_dic


def mkdirs(newdir):
    try: os.makedirs(newdir)
    except OSError as err:
        # Raise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


# Generates transcription file without allophonic variations

def generate_trans_SAE(trans_complete):

    complete_text = open(trans_complete)
    pruned_text = open("transcriptionsSAE.txt","w")

    d = [('Th/', ''), ('Kh/', ''), ('Ph/', ''), ('AX', 'AH0'), ('/DX', '')]

    s = complete_text.read()
    for i,o in d:
        s = s.replace(i,o)
    pruned_text.write(s)

    complete_text.close()
    pruned_text.close()

    return pruned_text

# Function that reads the output of gop-dnn and returns the
# phone alignments

def get_gop_alignments(path_filename, phone_pure_dict):

    output = []
    print(path_filename)
#    embed()
    for line in open(path_filename).readlines():
        print(line)
        l=line.split()

        if len(l) < 2:
            print("Invalid line")
        else:
            logid = l[0].replace("'", "")
            data = l[1:]
            i = 0
            phones = []
            gops = []
            phones_name = []
            while i < len(data):
                if data[i] == "[":
                    phone = int(data[i+1])
                    phone_name = phone_pure_dict[phone]

                    if phone_name not in ('SIL', 'sil', 'sp', 'spn', 'SP', 'SPN'):
                        gop = float(data[i+2])
                        phones.append(phone)
                        gops.append(gop)
                        phones_name.append(phone_name)


                    i = i + 4

            output.append({'logid': str(logid),
                           'phones':phones_name,
                           'gops':gops})

    df_phones = pd.DataFrame(output).set_index("logid")

    return df_phones


# Function that matches labels and phones from manual annotations with phones and scores from gop-dnn.
# This is a necessary step because manual annotations based on the forced alignments
# do not always coincide with gop's alignments.
# Note that whenever a phone is missing in the gop alignment a "?" is added to discard the corresponding label
# and, whenever a '0' (deletion) is present in the manual annotation, the gop score is discarded.


def match_labels2gop(logid, trans_zero, trans_manual, trans_auto, labels, gop_scores):

    rows = []
    j = 0
    position = 1

    # Agarra la transcripción manual que levanta de
    for i in range(0, len(trans_manual)):

        label = 0
        if(labels[i] == '+'):
            label = 1

        phone_manual = trans_manual[i]
        phone_zero = trans_zero[i]

        if j > len(trans_auto)-1:
            raise Exception("Index out of range")
        
        phone_automatic = trans_auto[j]
        rows.append([logid, phone_automatic, label, gop_scores[j], phone_manual, position])
        position += 1
        
        j = j + 1

    columns = ['logid', 'phone_automatic', 'label', 'gop_scores', 'phone_manual', 'position']
    df = pd.DataFrame(rows, columns=columns)
    return df

def get_reference(file):
    reference = []
    annot_manual = []
    labels = []
    i = 0
    for line in open(file).readlines():
        l=line.split()
        reference.append(l[1])
        annot_manual.append(l[2])
        labels.append(l[3])

        i += 1

    return reference, annot_manual, labels




def main(config_dict):
    reference_transcriptions_path = config_dict['reference-trans-path']
    utterance_list_path           = config_dict['utterance-list-path']
    output_dir                    = config_dict['eval-dir']
    output_filename               = config_dict['eval-filename']
    gop_path                      = config_dict['full-gop-score-path']
    phones_pure_path              = config_dict['kaldi-phones-pure-path']
    labels_dir_path               = config_dict['labels-dir-path']

    # Code that generates a pickle with useful data to analyze.
    # The outpul will be used to compute ROCs, AUCs and EERs.

    phone_pure_dict = phones2dic(phones_pure_path)
    gop_alignments = get_gop_alignments(gop_path, phone_pure_dict)

    utterance_list = generate_utterance_list_from_path(utterance_list_path) 
    trans_dict = get_reference_from_system_alignments(reference_transcriptions_path, labels_dir_path, gop_alignments, utterance_list)

    # Now, iterate over utterances
    output = []
    for utterance in utterance_list:
        gop_scores = gop_alignments.loc[utterance].gops

        annot_manual        = trans_dict[utterance]["trans_manual"]
        annot_kaldi         = trans_dict[utterance]["trans_auto"]
        trans_reff_complete = trans_dict[utterance]["best_ref_auto"]
        labels              = trans_dict[utterance]["labels"]
        trans_zero          = trans_dict[utterance]["best_ref_auto_zero"]

        df = match_labels2gop(utterance, trans_zero, annot_manual, annot_kaldi, labels, gop_scores)
        output.append(df)


    df_trans_match = pd.concat(output).set_index('logid')

    #Export file containing data for evaluation

    joblib.dump(df_trans_match, output_dir + output_filename)


