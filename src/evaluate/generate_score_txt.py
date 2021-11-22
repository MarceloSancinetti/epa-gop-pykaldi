import os
import glob
from pathlib import Path

import torchaudio
import torch
import torch.optim as optim

from src.utils.finetuning_utils import *
#from utils import *
from src.train.dataset import *

from torch.optim.swa_utils import AveragedModel

from src.pytorch_models.pytorch_models import *

from IPython import embed

import argparse

def removeSymbols(str, symbols):
    for symbol in symbols:
        str = str.replace(symbol,'')
    return str

def get_alignments(alignments_path):
    alignments_dict = {}

    for l in open(alignments_path, 'r').readlines():
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


def generate_scores_for_sample(phone_times, frame_level_scores):
    scores = []
    #Iterate over phone transcription and calculate score for each phome
    for phone_name, start_time, end_time in phone_times:
        #Do not score SIL phones
        if phone_name == 'SIL':
            continue

        #Check if the phone was uttered
        if start_time != end_time:
            current_phone_score = get_phone_score_from_frame_scores(frame_level_scores, 
                                                                    start_time, end_time, 'mean')
        #Use fixed negative score for deletions
        else:
            current_phone_score = -1000
        scores.append((phone_name, current_phone_score))
    return scores

def generate_scores_for_testset(model, testloader):
    print('Generating scores for testset')
    scores = {}
    for i, batch in enumerate(testloader, 0):       
        print('Batch ' + str(i+1) + '/' + str(len(testloader)))
        logids      = unpack_logids_from_batch(batch)
        features    = unpack_features_from_batch(batch)
        labels      = unpack_labels_from_batch(batch)
        phone_times = unpack_phone_times_from_batch(batch)
        outputs     = (-1) * model(features)

        frame_level_scores = get_scores_for_canonic_phones(outputs, labels)
        for i,logid in enumerate(logids):
            current_sample_scores = generate_scores_for_sample(phone_times[i], frame_level_scores[i])
            scores[logid] = current_sample_scores
    return scores

def log_sample_scores_to_txt(logid, scores, score_log_fh, phone_dict):
    score_log_fh.write(logid + ' ')
    for phone_name, score in scores:
        phone_number = phone_dict[phone_name]
        score_log_fh.write( '[ ' + str(phone_number) + ' ' + str(score)  + ' ] ')
    score_log_fh.write('\n')

def log_testset_scores_to_txt(scores, score_log_fh, phone_dict):
    print('Writing scores to .txt')
    for logid, sample_score in scores.items():
        log_sample_scores_to_txt(logid, sample_score, score_log_fh, phone_dict)

def main(config_dict):

    state_dict_dir      = config_dict['state-dict-dir']
    model_name          = config_dict['model-name']
    sample_list         = config_dict['utterance-list-path']
    phone_list_path     = config_dict['phones-list-path']
    labels_dir          = config_dict['auto-labels-dir-path']
    gop_txt_dir         = config_dict['gop-scores-dir']
    gop_txt_name        = config_dict['gop-txt-name']
    features_path       = config_dict['features-path']
    conf_path           = config_dict['features-conf-path']
    device_name         = config_dict['device']
    batchnorm           = config_dict['batchnorm']

    testset = EpaDB(sample_list, phone_list_path, labels_dir, features_path, conf_path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2,
                                          shuffle=False, num_workers=0, collate_fn=collate_fn_padd)

    phone_count = testset.phone_count()

    #Get acoustic model to test
    model = FTDNN(out_dim=phone_count, device_name=device_name, batchnorm=batchnorm)
    if model_name.split("_")[-1] == "swa":
        model = AveragedModel(model)
    
    model.eval()
    state_dict = torch.load(state_dict_dir + '/' + model_name + '.pth')
    model.load_state_dict(state_dict['model_state_dict'])

    phone_dict = testset._phone_sym2int_dict

    scores = generate_scores_for_testset(model, testloader)
    score_log_fh = open(gop_txt_dir+ '/' + gop_txt_name, 'w+')
    log_testset_scores_to_txt(scores, score_log_fh, phone_dict)

