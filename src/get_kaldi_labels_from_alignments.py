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


# Function that reads the output from pykaldi aligner and returns the
# phone alignments

def get_kaldi_alignments(path_filename):

    output = []
    unwanted_characters = '[\[\]()\'\",]'
    print(path_filename)
    for line in open(path_filename).readlines():
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


def get_reference(file):
    reference = []
    annot_manual = []
    labels = []
    start_times = []
    end_times = []
    
    for line in open(file).readlines():
        l=line.split()
        reference.append(l[1])
        annot_manual.append(l[2])
        labels.append(l[3])
    	

    return reference, annot_manual, labels

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
                #embed()
                pass
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

def get_times(labels_file, kaldi_alignments, utterance, times_source):
    
    if times_source == 'kaldi':
        start_times = kaldi_alignments.loc[utterance].start_times
        end_times = kaldi_alignments.loc[utterance].end_times
    
    if times_source == 'manual':
        start_times = []
        end_times = []
        
        for line in open(labels_file).readlines():
            l=line.split()
            start_times.append(l[4])
            end_times.append(l[5])

    return start_times, end_times


#TODO
def assign_kaldi_labels(kaldi_alignments, utterance, annot_manual):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcription-file', dest='transcriptions', help='File with reference phonetic transcriptions of each of the phrases', default=None)
    parser.add_argument('--utterance-list', dest='utterance_list', help='File with utt list', default=None)
    parser.add_argument('--reference-file', dest='reference_path', help='', default=None)
    parser.add_argument('--alignment-file', dest='align_path', help='', default=None)
    parser.add_argument('--output', dest='output_dir', help='Output directory for labels', default=None)
    parser.add_argument('--labels', dest='label_source', help='Method to use for labels, \'kaldi\' or \'ref\' ', default=None)
    parser.add_argument('--target', dest='target_source', help='Transcription to use as target column, \'kaldi\' or \'ref\' ', default=None)
    parser.add_argument('--times', dest='times_source', help='Annotation to use for start/end times, \'kaldi\' or \'manual\' ', default=None)


    args = parser.parse_args()

    # Code that generates a pickle with useful data to analyze.
    # The outpul will be used to compute ROCs, AUCs and EERs.

    output = []
    output_tmp  = []

    trans_dict_complete, trans_dict_clean_complete, sent_dict_complete = generate_dict_from_transcripctions(args.transcriptions)
    generate_trans_SAE(args.transcriptions)

    kaldi_alignments = get_kaldi_alignments(args.align_path)

    utterance_list = []
    utt_list_fh = open(args.utterance_list, 'r')
    for line in utt_list_fh.readlines():
        logid = line.split(' ')[0]
        utterance_list.append(logid)


    # Now, iterate over utterances
    for utterance in utterance_list:

        spk, sent = utterance.split("_")

        file = "%s/%s/%s/%s.txt"%(args.reference_path, spk, "labels", utterance) #Labels file for current utterance
        print("----------------------------------------------------------------------------------------")
        print("Speaker %s, sentence %s: %s (File: %s)"%(spk, sent, " ".join(sent_dict_complete[sent]), file))
        
        #Get phone list from manual annotation 
        trans_reff_complete, annot_manual, labels_ref = get_reference(file)

        start_times, end_times = get_times(file, kaldi_alignments, utterance, args.times_source)

        if args.label_source == 'kaldi':
            labels = assign_kaldi_labels(kaldi_alignments, utterance, annot_manual)
        if args.label_source == 'ref':
            labels = labels_ref



        if utterance in kaldi_alignments.index.values:
            annot_kaldi = kaldi_alignments.loc[utterance].phones
        else:
            raise Exception("WARNING: Missing alignment for "+ utterance)



        # Find the transcription for this sentence that best matches the annotation

        best_trans = -1
        best_trans_corr = 0

        for trans_idx, trans in enumerate(trans_dict_clean_complete[sent]):
            if(len(trans) == len(annot_kaldi)):
                num_correct = np.sum([t==a for t, a in np.c_[trans,annot_kaldi]])
                if num_correct > best_trans_corr:
                    best_trans_corr = num_correct
                    best_trans = trans_idx


        if best_trans != -1:


            trans      = trans_dict_clean_complete[sent][best_trans]
            trans_zero = trans_dict_complete[sent][best_trans]



            print("TRANS_REFF:           %s (chosen out of %d transcriptions)"%(phonelist2str(trans), len(trans_dict_clean_complete[sent])))
            print("TRANS_KALDI:          "+phonelist2str(annot_kaldi))
            print("LABEL:                "+phonelist2str(labels))
            print("TRANS_ZERO:           "+phonelist2str(trans_zero))
            print("TRANS_MANUAL:         "+phonelist2str(annot_manual))
            print("TRANS_REFF_COMPLETE:  "+phonelist2str(trans_reff_complete))
            print("TRANS_WITHOUT_ZERO:   "+phonelist2str(trans))

            if args.target_source == 'kaldi':
                target_column = annot_kaldi
                if len(target_column) != len(annot_manual):
                    try:
                    	_, annot_manual, labels = remove_deletion_lines(trans_zero, annot_manual, labels)
                    except IndexError as e:
                    	embed()
                if len(target_column) != len(annot_manual):
                    annot_manual, target_column, labels, start_times, end_times = remove_deletion_lines(annot_manual, target_column, labels, remove_times=True, start_times=start_times, end_times=end_times)
            if args.target_source == 'ref':
                target_column = trans_reff_complete
                target_column, annot_manual, labels, start_times, end_times = remove_deletion_lines(target_column, annot_manual, labels, start_times, end_times)
       
            #target_column, annot_manual, labels, start_times, end_times = remove_deletion_lines(target_column, annot_manual, labels, start_times, end_times)

            #if len(labels) < len(annot_kaldi):
                #annot_kaldi = remove_non_labeled_phones_from_kaldi_annotation(annot_kaldi, annot_manual)  
            #    raise Exception('Kaldi annotaton is longer than manual annotation. Logid: ' + utterance)

            

            outdir  = "%s/labels_with_kaldi_phones/%s" % (args.output_dir, spk)
            outfile = "%s/%s.txt" % (outdir, utterance)
            mkdirs(outdir)
            try:
                np.savetxt(outfile, np.c_[np.arange(len(target_column)), target_column, annot_manual, labels, start_times, end_times], fmt=utterance+"_%s %s %s %s %s %s")
            except ValueError as e:
                embed()

        else:

            print(trans_dict_clean_complete[sent])
            print(phones)
            print(annot_kaldi)
            raise Exception("WARNING: %s does not match with transcription"%(spk))




