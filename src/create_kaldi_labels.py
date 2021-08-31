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
from finetuning_utils import *

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

def add_phone_count(phone_count_dict, transcription, as_int=False, separate_pos_neg=False, labels=None):
    global phone_sym2int

    for i, phone in enumerate(transcription):
        if as_int:
            phone = phone_sym2int[phone]
        if separate_pos_neg:
            phone = phone + labels[i]
        
        if phone in phone_count_dict:
            phone_count_dict[phone] = phone_count_dict[phone] + 1
        else:
            phone_count_dict[phone] = 1
    return phone_count_dict

def create_phone_weight_yaml(phone_weights_path, phone_count_dict, class_count_dict):
    global phone_sym2int, phone_int2sym
    

    phone_weights_fh = open(phone_weights_path, "w+")
    phone_weights_fh.write("---\n")

    for phone in sorted(phone_count_dict.keys()):
        phone_sym = phone_int2sym[phone]
        minority_occurences = min(class_count_dict[phone_sym + '-'], class_count_dict[phone_sym + '+'])
        occurrences = phone_count_dict[phone]

        if occurrences < 100 or minority_occurences < 30:
            phone_weights_fh.write("  phone" + str(phone) + ":    " + str(0) + "\n")
        else:
            total_occurrences = sum(phone_count_dict.values())
            phone_count = len(phone_count_dict.keys())
            weight = occurrences / total_occurrences * phone_count * 2
            phone_weights_fh.write("  phone" + str(phone) + ":    " + str(weight) + "\n")

def create_phone_class_count_list(class_count_dict):
    class_counts_fh = open("phone_class_counts.yaml", "w+")
    class_counts_fh.write("---\n")
    for phone in sorted(class_count_dict.keys()):
        occurrences = class_count_dict[phone]
        class_counts_fh.write("  " + str(phone) + ":    " + str(occurrences) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-transcriptions-path', dest='reference_transcriptions_path', help='Path to file with reference phonetic transcriptions of each of the phrases', default=None)
    parser.add_argument('--utterance-list-path', dest='utterance_list_path', help='File with utt list', default=None)
    parser.add_argument('--labels-dir-path', dest='labels_dir_path', help='Path to EpaDB labels directory', default=None)
    parser.add_argument('--alignments-path', dest='align_path', help='Path to alignment output file', default=None)
    parser.add_argument('--output-dir-path', dest='output_dir_path', help='Path to output directory for labels', default=None)
    parser.add_argument('--phones-list-path', dest='phone_list_path', help='Path to phone list', default=None)
    parser.add_argument('--phone-weights-path', dest='phone_weights_path', help='Path to .yaml containing weights for phone-level loss', default=None)
    parser.add_argument('--phone-count', dest='phone_count', help='Amount of phones in the phone set', default=None)

    args = parser.parse_args()
    global phone_count

    reference_transcriptions_path = args.reference_transcriptions_path
    utterance_list_path           = args.utterance_list_path
    labels_dir_path               = args.labels_dir_path
    phone_count                   = int(args.phone_count)
    create_phone_count_yamls      = False

    kaldi_alignments = get_kaldi_alignments(args.align_path)
    utterance_list = generate_utterance_list_from_path(utterance_list_path) 
    trans_dict = get_reference_for_system_alignments(reference_transcriptions_path, labels_dir_path, kaldi_alignments, utterance_list)

    global phone_sym2int, phone_int2sym
    phone_sym2int = get_phone_symbol_to_int_dict(args.phone_list_path)
    phone_int2sym = get_phone_int_to_symbol_dict(args.phone_list_path)

    phone_int_count_dict = {phone:0 for phone in range(phone_count)}
    phone_sym_count_dict = {phone_int2sym[phone_int]+'+':0 for phone_int in range(phone_count)}
    phone_sym_count_dict.update({phone_int2sym[phone_int]+'-':0 for phone_int in range(phone_count)})

    for utterance in utterance_list:
        spk, sent = utterance.split("_")

        start_times, end_times = get_times(kaldi_alignments, utterance)
        target_column, trans_manual, labels, start_times, end_times = match_trans_lengths(trans_dict[utterance], start_times, end_times)       

        #Add occurrences of each phone to the phone count dict
        phone_int_count_dict = add_phone_count(phone_int_count_dict, target_column, as_int=True)
        phone_sym_count_dict = add_phone_count(phone_sym_count_dict, target_column, separate_pos_neg=True, labels=labels)


        #outdir  = "%s/labels_with_kaldi_phones/%s/labels" % (args.output_dir_path, spk)
        outdir  = "%s/%s/labels" % (args.output_dir_path, spk)
        outfile = "%s/%s.txt" % (outdir, utterance)
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        try:
            np.savetxt(outfile, np.c_[np.arange(len(target_column)), target_column, trans_manual, labels, start_times, end_times], fmt=utterance+"_%s %s %s %s %s %s")
        except ValueError as e:
            embed()

    if create_phone_count_yamls:
        create_phone_class_count_list(phone_sym_count_dict)
        create_phone_weight_yaml(args.phone_weights_path, phone_int_count_dict, phone_sym_count_dict)    
