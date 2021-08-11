import os, errno, re
import os.path as path
from os import remove
import numpy as np
from scipy.stats.stats import pearsonr
from IPython import embed
import pandas as pd
import joblib
import shutil
import argparse
import glob
from reference_utils import *

def log_problematic_utterance(utterance):
    pu_fh = open("problematic_utterances", "a+")
    pu_fh.write(utterance + '\n')


#This function discards positions from manual transcription and labels where the automatic transcription reference is
#silent, i.e there is no automatic transcription for said position so that the lengths of all sequences match
def match_trans_lengths(trans_dict, start_times, end_times):
    trans_auto               = trans_dict['trans_auto']
    trans_manual             = trans_dict['trans_manual']
    best_ref_auto_zero       = trans_dict['best_ref_auto_zero']
    labels                   = trans_dict['labels']

    if '0' in best_ref_auto_zero:
        _, trans_manual, labels = remove_deletion_lines(best_ref_auto_zero, trans_manual, labels)

    #Ojo con esto, no deberia hacer falta
    if len(trans_auto) != len(trans_manual):
        log_problematic_utterance(utterance)
        trans_manual, trans_auto, labels, start_times, end_times = remove_deletion_lines(trans_manual, trans_auto, labels, 
                                                       remove_times=True, start_times=start_times, end_times=end_times)

    return trans_auto, trans_manual, labels, start_times, end_times

# Function that reads the output from pykaldi aligner and returns the
# phone alignments

def get_kaldi_alignments(alignment_file_path):

    output = []
    unwanted_characters = '[\[\]()\'\",]'
    print(alignment_file_path)
    for line in open(alignment_file_path).readlines():
        l=line.split()

        if 'phones' == l[1]:
            print(l)
            logid = l[0]
            data = l[2:]
            i = 0
            phones = []
            start_times = []
            end_times = []
            while i < len(data):
                phone = re.sub(unwanted_characters, '', data[i])
                #Turn phone into pure phone (i.e. remove _context)
                if '_' in phone:
                    phone = phone[:-2]
                if phone[-1] in ['1', '0', '2']:
                    phone = phone[:-1]

                if phone not in ['sil', '[key]', 'sp', '', 'SIL', '[KEY]', 'SP']:
                    phones.append(phone)
                    start_time  = re.sub(unwanted_characters, '', data[i+1])
                    duration    = re.sub(unwanted_characters, '', data[i+2])
                    start_times.append(start_time)
                    end_times.append(str(int(start_time) + int(duration)))
                i = i + 3


            output.append({'logid': str(logid),
                           'phones' :phones,
                           'start_times' :start_times,
                           'end_times'   :end_times })

    df_phones = pd.DataFrame(output).set_index("logid")

    return df_phones

def remove_deletion_lines_with_times(trans1, trans2, labels, start_times, end_times):
    clean_trans1 = []
    clean_trans2 = []
    clean_labels = []
    clean_start_times = []
    clean_end_times = []
    for i, phone in enumerate(trans1):
        if phone != '0':
            try:
                clean_trans1.append(phone)
                clean_trans2.append(trans2[i])            
                clean_labels.append(labels[i])
                clean_start_times.append(start_times[i])
                clean_end_times.append(end_times[i])
            except IndexError as e:
                log_problematic_utterance(utterance)
                embed()
    return clean_trans1, clean_trans2, clean_labels, clean_start_times, clean_end_times

def remove_deletion_lines(trans1, trans2, labels, remove_times=False, start_times=None, end_times=None):
    #Times should be provided iff their deletion lines should be removed
    if remove_times and (start_times == None or end_times == None):
        raise Exception('remove_times is True but start or end times are missing')
    if not remove_times and (start_times != None or end_times != None):
        raise Exception('remove_times is False but start or end times were given')
    
    if remove_times:
       return remove_deletion_lines_with_times(trans1, trans2, labels, start_times, end_times)
    else:
        #If start or end times are not needed, dummy times are passed
        trans1, trans2, labels, _, _ = remove_deletion_lines_with_times(trans1, trans2, labels, range(len(trans1)), range(len(trans1)))  
        return trans1, trans2, labels

def get_times(kaldi_alignments, utterance):
    
    start_times = kaldi_alignments.loc[utterance].start_times
    end_times = kaldi_alignments.loc[utterance].end_times
    
    return start_times, end_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-transcriptions-path', dest='reference_transcriptions_path', help='Path to file with reference phonetic transcriptions of each of the phrases', default=None)
    parser.add_argument('--utterance-list-path', dest='utterance_list_path', help='File with utt list', default=None)
    parser.add_argument('--labels-dir-path', dest='labels_dir_path', help='Path to EpaDB labels directory', default=None)
    parser.add_argument('--alignments-path', dest='align_path', help='Path to alignment output file', default=None)
    parser.add_argument('--output-dir-path', dest='output_dir_path', help='Path to output directory for labels', default=None)

    args = parser.parse_args()

    reference_transcriptions_path = args.reference_transcriptions_path
    utterance_list_path           = args.utterance_list_path
    labels_dir_path               = args.labels_dir_path

    kaldi_alignments = get_kaldi_alignments(args.align_path)
    utterance_list = generate_utterance_list_from_path(utterance_list_path) 
    trans_dict = get_reference_for_system_alignments(reference_transcriptions_path, labels_dir_path, kaldi_alignments, utterance_list)

    for utterance in utterance_list:
        spk, sent = utterance.split("_")

        start_times, end_times = get_times(kaldi_alignments, utterance)
        target_column, trans_manual, labels, start_times, end_times = match_trans_lengths(trans_dict[utterance], start_times, end_times)       

        #outdir  = "%s/labels_with_kaldi_phones/%s/labels" % (args.output_dir_path, spk)
        outdir  = "%s/%s/labels" % (args.output_dir_path, spk)
        outfile = "%s/%s.txt" % (outdir, utterance)
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        try:
            np.savetxt(outfile, np.c_[np.arange(len(target_column)), target_column, trans_manual, labels, start_times, end_times], fmt=utterance+"_%s %s %s %s %s %s")
        except ValueError as e:
            embed()
