import os
import glob
from pathlib import Path

import torchaudio
import torch
import torch.optim as optim

from finetuning_utils import *
from utils import *
from dataset import *

from pytorch_models import *

from IPython import embed

import argparse

def removeSymbols(str, symbols):
    for symbol in symbols:
        str = str.replace(symbol,'')
    return str

def get_alignments(alignments_dir_path):
    alignments_dict = {}

    for l in open(alignments_dir_path + "align_output", 'r').readlines():
        l=l.split()
        #Get phones alignments
        if len(l) > 3 and l[1] == 'phones':
            logid = l[0]
            alignment = []
            alignments_dict[logid] = {}
            for i in range(2, len(l),3):
                current_phone =     removeSymbols(l[i],  ['[',']',',',')','(','\''])
                start_time    = int(removeSymbols(l[i+1],['[',']',',',')','(','\'']))
                duration      = int(removeSymbols(l[i+2],['[',']',',',')','(','\'']))
                end_time      = start_time + duration
                alignment.append((current_phone, start_time, end_time))
            alignments_dict[logid] = alignment
    return alignments_dict


def generate_score_txt(model, testloader, score_file_name, phone_dict, alignments_dir_path):

    print('Writing scores to .txt')
    alignments = get_alignments(alignments_dir_path)
    for i, batch in enumerate(testloader, 0):
        score_log_fh = open(score_file_name, "a+")
        print('Batch ' + str(i+1) + '/' + str(len(testloader)))

        logids = unpack_logids_from_batch(batch)
        features = unpack_features_from_batch(batch)
        labels = unpack_labels_from_batch(batch)

        outputs = (-1) * model(features)

        frame_level_scores = get_scores_for_canonic_phones(outputs, labels)
        #Iterate over samples in the test batch
        for i, logid in enumerate(logids):
            score_log_fh.write(logid + ' ')
            #Iterate over phones in the annotation for the current sample
            for phone_name, start_time, end_time in alignments[logid]:
                #Check if the phone was uttered
                if start_time != end_time:
                    #Log the score for the current frame in the annotation
                    try:
                        log_phone_number_and_score(score_log_fh, labels[i], frame_level_scores[i],
                                                   start_time, end_time, 'mean')
                    except ValueError as e:
                        embed()
                else:
                    try:
                        phone_number = phone_dict[phone_name] + 3 #This +3 is here to take into account eps, sil, spn 
                    except KeyError as e:
                        embed()
                    score_log_fh.write( '[ ' + str(phone_number) + ' -1000'  + ' ] ')
            score_log_fh.write('\n')
    
        score_log_fh.close()           


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict-dir', dest='state_dict_dir', help='Directory to saved state dicts in .pth', default=None)
    parser.add_argument('--model-name', dest='model_name', help='Model name (usually the name of the wandb run that generated the .pth file)', default=None)
    parser.add_argument('--epa-root', dest='epa_root_path', help='EpaDB root directory', default=None)
    parser.add_argument('--sample-list', dest='sample_list_path', help='Path to list of samples to test on', default=None)
    parser.add_argument('--phone-list', dest='phone_list_path', help='Path to phone list', default=None)
    parser.add_argument('--labels-dir', dest='labels_dir', help='Directory where labels are found', default=None)     
    parser.add_argument('--gop-txt-dir', dest='gop_txt_dir', help='Directory to save generated scores', default=None)
    parser.add_argument('--features-path', dest='features_path', help='Path to features directory', default=None)
    parser.add_argument('--conf-path', dest='conf_path', help='Path to config directory used in feature extraction', default=None)
    args = parser.parse_args()

    state_dict_dir = args.state_dict_dir
    model_name = args.model_name
    epa_root_path = args.epa_root_path
    sample_list = args.sample_list_path
    phone_list_path = args.phone_list_path
    labels_dir = args.labels_dir
    gop_txt_dir = args.gop_txt_dir
    features_path = args.features_path
    conf_path = args.conf_path

    testset = EpaDB(epa_root_path, sample_list, phone_list_path, labels_dir, features_path, conf_path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2,
                                          shuffle=False, num_workers=2, collate_fn=collate_fn_padd)

    phone_count = testset.phone_count()

    #Get acoustic model to test
    model = FTDNN(out_dim=phone_count)
    state_dict = torch.load(state_dict_dir + '/' + model_name + '.pth')
    model.load_state_dict(state_dict['model_state_dict'])

    phone_dict = testset._pure_phone_dict

    generate_score_txt(model, testloader, gop_txt_dir+ '/' +'gop-'+model_name+'.txt', phone_dict)


if __name__ == '__main__':
    main()