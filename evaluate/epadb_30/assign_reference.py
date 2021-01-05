#!/usr/bin/env python
# coding: utf-8

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

def phonelist2str(phones):
    return " ".join(["%3s"%p for p in phones])


def mkdirs(newdir):
    try: os.makedirs(newdir)
    except OSError as err:
        # Raise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


# Generates transcription file without allophonic variations



# Function that reads transcriptions files and loads them to
# a series of useful dictionaries

def generate_dict_from_transcripctions(transcriptions):

    trans_dict = dict()
    trans_dict_clean = dict()
    sent_dict = dict()

    # Read transcription file
    for line in open(transcriptions,'r'):

        fields = line.strip().split()

        if len(fields) <= 2:
            continue

        sent = fields[1].strip(":")

        if fields[0] == "TEXT":
            sent_dict[sent] = fields[2:]

        if fields[0] != "TRANSCRIPTION":
            continue

        if sent not in trans_dict_clean:

            # Before loading the first transcription for a sentence,
            # create an entry for it in the dict. The entry will be a
            # list of lists. One list for each possible transcription
            # for that sentence.

            trans_dict[sent] = list()
            trans_dict_clean[sent] = list()

        trans = [[]]
        for i in range(2, len(fields)):
            phones = fields[i].split("/")

            # Reproduce the transcriptions up to now as many times as
            # the number of phone variations in this slot. Then, append
            # one variation to each copy.

            trans_new = []
            for p in phones:
                for t in trans:
                    t_tmp = t + [p.strip()]
                    trans_new.append(t_tmp)
            trans = trans_new

        trans_dict[sent] += trans

    for sent, trans in trans_dict.items():
        trans_clean_new = []
        for t in trans:
            trans_clean_new.append([x for x in t if x != '0'])

        if sent not in trans_dict_clean:
            trans_dict_clean[sent] = list()

        trans_dict_clean[sent] += trans_clean_new

    return trans_dict, trans_dict_clean, sent_dict






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcription-file', dest='transcriptions', help='File with reference phonetic transcriptions of each of the phrases', default=None)
    parser.add_argument('--annotation-dir', dest='annotation_dir', help='Directory with textgrid files with annotations', default=None)
    parser.add_argument('--output-dir', dest='output_dir', help='Output dir', default=None)

    args = parser.parse_args()

    # Code that generates a pickle with useful data to analyze.
    # The outpul will be used to compute ROCs, AUCs and EERs.

    output = []


    trans_dict_complete, trans_dict_clean_complete, sent_dict_complete = generate_dict_from_transcripctions(args.transcriptions)


    utterance_list = [re.sub('.TextGrid','', re.sub('.*\/','',s)) for s in glob.glob("*/%s/*.TextGrid"%args.annotation_dir)]
    # Now, iterate over utterances
    for utterance in utterance_list:

        #embed()
        spk, sent = utterance.split("_")
        tgfile = "%s/%s/%s.TextGrid"%(spk, args.annotation_dir, utterance) #TextGrid file for current utterance

        print("----------------------------------------------------------------------------------------")
        print("Speaker %s, sentence %s: %s (File: %s)"%(spk, sent, " ".join(sent_dict_complete[sent]), tgfile))

        try:
            tg = textgrids.TextGrid(tgfile)
        except:

            raise Exception("Bad textgrid file %s"%tgfile)

        if len(tg) < 3:

            raise Exception("WARNING: File %s does not have an annotation or a score tier\n"%tgfile)


        #Get phone list from manual annotation in current textgrid

        annot_manual = []

        for i in tg['annotation']:
            p = i.text.strip()
            start = i.xmin
            end = i.xmax
            if p not in ['sil', '', 'sp', 'None']:
                if p[-1] not in ['0','1', '2']:
                    annot_manual += [p]
                else:
                    if p == '0' or p[-1] not in ['0','1', '2']:
                    #if p[-1] not in ['0','1', '2']:
                        annot_manual += [p]
                    else:
                        annot_manual += [p]  if p == 'AH0' else [p[:-1]]

        # Find the transcription for this sentence that best matches the annotation

        best_trans1 = -1
        best_trans_corr = 0


        best_trans1 = -1
        best_trans_corr = 0
        for trans_idx, trans1 in enumerate(trans_dict_complete[sent]):
            if(len(trans1) == len(annot_manual)):
                num_correct = np.sum([t==a for t, a in np.c_[trans1,annot_manual]])
                if num_correct > best_trans_corr:
                    best_trans_corr = num_correct
                    best_trans1 = trans_idx


        if best_trans1 != -1:

            trans_reff_complete = trans_dict_complete[sent][best_trans1]
            labels = np.array(['+' if t==a else '-' for t, a in np.c_[trans_reff_complete,annot_manual]])

            print("MANUAL_ANNOTATION:         "+phonelist2str(annot_manual))
            print("REFFERENCE_TRANSCRIPTION:  "+phonelist2str(trans_reff_complete))
            print("LABELS:                    "+phonelist2str(labels))

            outdir  = "%s/%s" % (spk, "labels")
            outfile = "%s/%s.txt" % (outdir, utterance)
            mkdirs(outdir)
            np.savetxt(outfile, np.c_[np.arange(len(annot_manual)), trans_reff_complete, annot_manual, labels], fmt=utterance+"_%s %s %s %s")


        else:

            raise Exception("WARNING: %s %s does not match with transcription"%(spk, sent))
