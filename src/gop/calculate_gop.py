import pandas as pd
import pickle
import os
from utils import *
from gop import *
from scipy.special import softmax
from kaldiio import ReadHelper
import tqdm
import argparse


def prepare_dataframes(libri_phones_path, libri_phones_to_pure_int_path, 
                       libri_phones_pure_path, libri_final_mdl_path, gop_dir, alignments_dir_path):

    if not os.path.exists(gop_dir + '/phones_pure_epa.pickle'):
        generate_df_phones_pure(libri_phones_path, libri_phones_to_pure_int_path, 
                                libri_phones_pure_path, libri_final_mdl_path, gop_dir)

    if not os.path.exists(gop_dir + '/alignments.pickle'):
        generate_df_alignments(gop_dir, alignments_dir_path)


    df_phones_pure = pd.read_pickle(gop_dir + '/phones_pure_epa.pickle')
    df_phones_pure = df_phones_pure.reset_index()

    df_alignments = pd.read_pickle(gop_dir + '/alignments.pickle')
    df_alignments = pd.DataFrame.from_dict(df_alignments)

    return df_phones_pure, df_alignments

def pad_loglikes(loglikes):
    max_frames = max([x.shape[0] for x in loglikes])
    padded_loglikes = [np.pad(x, ((0, max_frames - len(x)), (0,0)), 'constant', 
                       constant_values=(0, 0) ) for x in tqdm.tqdm(loglikes)]
    return padded_loglikes



def compute_gop(gop_dir, df_phones_pure, df_alignments, loglikes_path, utterance_list_path, batch_size):

    gop = {}
    gop_dict = {}
    loglikes_all_utts = []
    with ReadHelper('ark:' + loglikes_path) as reader:
        for key, loglikes in tqdm.tqdm(reader):
            loglikes = softmax(np.array(loglikes), axis=1) #Apply softmax before computing
            loglikes_all_utts.append(loglikes)

    df_scores = df_alignments.transpose()
    df_scores['p'] = np.array(loglikes_all_utts, dtype=object)
    padded_loglikes = pad_loglikes(df_scores['p'])
    df_scores['p'] = padded_loglikes

    for i in range(0, len(df_scores), batch_size):
        df_scores_batch = df_scores.iloc[i:i+batch_size]
        loglikes_batch  = loglikes_all_utts[i:i+batch_size]
        gop_dict = gop_robust_with_matrix(df_scores_batch, df_phones_pure, 6024, len(df_scores_batch), gop_dict)

    validate_gop_samples_with_utterance_list(gop_dict, utterance_list_path)

    with open(gop_dir + '/gop.pickle', 'wb') as handle:
        pickle.dump(gop_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_gop_as_text(gop_dir):

    gop_dict = pd.read_pickle(gop_dir + '/gop.pickle')


    gop_output_file = open(gop_dir + '/gop.txt', 'w+')

    for logid, score in gop_dict.items():
        phones = score['phones_pure']
        gop    = score['gop']
        
        if len(phones) != len(gop):
            raise Exception("Phones and gop list lengths do not match.")
        
        gop_output_file.write(logid)
        for i in range(len(phones)):
            gop_output_file.write(' [ ' + phones[i] + ' ' + str(gop[i]) + ' ] ')
        gop_output_file.write('\n')


def validate_gop_samples_with_utterance_list(gop_dict, utterance_list_path):
    utt_list_fh = open(utterance_list_path, "r")
    utt_set = set()
    
    for line in utt_list_fh.readlines():
        utt_set.add(line.split()[0])

    utts_in_gop = set(gop_dict.keys())

    if utts_in_gop != utt_set:
        raise Exception("ERROR: Keys in gop dictionary do not match utterance list.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--libri-phones-path', dest='libri_phones_path', help='Path to Librispeech phones.txt file', default=None)
    parser.add_argument('--libri-phones-to-pure-int-path', dest='libri_phones_to_pure_int_path', help='Path to Librispeech phones-to-pure-phone.int', default=None)
    parser.add_argument('--libri-phones-pure-path', dest='libri_phones_pure_path', help='Path to Librispeech phones-pure.txt', default=None)
    parser.add_argument('--transition-model-path', dest='libri_final_mdl_path', help='Path to Librispeech final.mdl transition model', default=None)
    parser.add_argument('--gop-dir', dest='gop_dir', help='Path GOP directory', default=None)
    parser.add_argument('--alignments-dir-path', dest='alignments_dir_path', help='Path to directory where align_output will be found', default=None)
    parser.add_argument('--loglikes-path', dest='loglikes_path', help='Path to loglikes.ark', default=None)     
    parser.add_argument('--utterance-list-path', dest='utterance_list_path', help='Path EpaDB samples list', default=None)     
    args = parser.parse_args()

    libri_phones_path             = args.libri_phones_path
    libri_phones_to_pure_int_path = args.libri_phones_to_pure_int_path
    libri_phones_pure_path        = args.libri_phones_pure_path 
    libri_final_mdl_path          = args.libri_final_mdl_path 
    gop_dir                       = args.gop_dir
    loglikes_path                 = args.loglikes_path
    alignments_dir_path           = args.alignments_dir_path

    df_phones_pure, df_alignments = prepare_dataframes(libri_phones_path, libri_phones_to_pure_int_path, libri_phones_pure_path,
                                                       libri_final_mdl_path, gop_dir, alignments_dir_path)

    compute_gop(gop_dir, df_phones_pure, df_alignments, loglikes_path, args.utterance_list_path, 128)

    save_gop_as_text(gop_dir)