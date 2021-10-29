import os
import glob
from pathlib import Path

import torchaudio
import torch
import torch.optim as optim

#from utils import *
#from dataset import *

from src.pytorch_models.pytorch_models import *

import wandb

from IPython import embed


def unpack_logids_from_batch(batch):
    return [item['speaker_id'] + '_' + item['utterance_id'] for item in batch]

def unpack_features_from_batch(batch):
    return torch.stack([item['features'] for item in batch])

def unpack_pos_labels_from_batch(batch):
    return torch.stack([item['pos_labels'] for item in batch])

def unpack_labels_from_batch(batch):
    return torch.stack([item['labels'] for item in batch])

def unpack_neg_labels_from_batch(batch):
    return torch.stack([item['neg_labels'] for item in batch])

def unpack_phone_times_from_batch(batch):
    return [item['phone_times'] for item in batch]

def collate_fn_padd(batch):
    '''
    Padds batch of variable length (both features and labels)
    '''
    ## padd
    batch_features   = [item['features'] for item in batch]
    batch_features   = torch.nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
    batch_pos_labels = [item['pos_labels'] for item in batch]
    batch_pos_labels = torch.nn.utils.rnn.pad_sequence(batch_pos_labels, batch_first=True, padding_value=-1)
    batch_neg_labels = [item['neg_labels'] for item in batch]
    batch_neg_labels = torch.nn.utils.rnn.pad_sequence(batch_neg_labels, batch_first=True, padding_value=-1)
    batch_labels     = [item['labels'] for item in batch]
    batch_labels     = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=0)
    for i in range(len(batch)):
        batch[i]['features']   = batch_features[i]
        batch[i]['pos_labels'] = batch_pos_labels[i]
        batch[i]['neg_labels'] = batch_neg_labels[i]
        batch[i]['labels']     = batch_labels[i]
    return batch

#Creates dictionary mapping phone symbol to integer given a phone list path
def get_phone_symbol_to_int_dict(phone_list_path):
    #Open file that contains list of pure phones
    phones_list_fh = open(phone_list_path, "r")

    phone_dict = {}
    #Get phone number for each phone
    for i, phone_pure_name in enumerate(phones_list_fh.readlines()):
        phone_dict[phone_pure_name.strip()] = i
    return phone_dict

#Creates dictionary mapping phone integer to symbol given a phone list path
def get_phone_int_to_symbol_dict(phone_list_path):
    #Open file that contains list of pure phones
    phones_list_fh = open(phone_list_path, "r")

    phone_dict = {}
    #Get phone number for each phone
    for i, phone_pure_name in enumerate(phones_list_fh.readlines()):
        phone_dict[i] = phone_pure_name.strip()
    return phone_dict

#The model outputs a score for each phone in each frame. This function extracts only the relevant scores,
#i.e the scores for the canonic phone in each frame based on the annotations.
#If a frame has no canonic phone (silence frame), the score is set to 0.
def get_scores_for_canonic_phones(outputs, labels):
    #Generate mask based on non-zero labels
    outputs_mask = torch.abs(labels)
    #Mask outputs and sum over phones to get a single value for the relevant phone in each frame
    outputs = outputs * outputs_mask
    outputs = torch.sum(outputs, dim=2)
    return outputs

#This function returns the non-zero relevant scores and 0/1 labels to calculate loss
def get_outputs_and_labels_for_loss(outputs, labels):
    outputs = get_scores_for_canonic_phones(outputs, labels)
    #Sum over phones to keep relevant label for each frame
    labels = torch.sum(labels, dim=2)
    #Remove labels == 0 (silence frames) in both labels and outputs
    outputs = outputs[labels != 0]
    labels = labels[labels != 0]
    #Turn 1s into 0s and -1s into 1s to pass the labels to loss_fn
    labels = labels - 1
    labels = torch.abs(labels / 2)    
    return outputs, labels


#Returns the canonic phone number at a given frame in the labels 
def get_phone_number_at_frame(labels, frame):
    #try: 
    res = labels[frame].nonzero().item()
    #except ValueError as e:
    #	embed()
    return res + 3 #This +3 is here because librispeech phones-pure starts with eps, sil, spn and these should be skipped 

#Collapses multiple frame level scores using sum or mean  
def get_phone_score_from_frame_scores(frame_level_scores, start_time, end_time, method):
    if   method == 'sum':
        return torch.sum(frame_level_scores[start_time:end_time]).item()
    elif method == 'mean':
        return torch.mean(frame_level_scores[start_time:end_time]).item()
    else:
        raise Exception('Unsupported frame score collapse method ' + method)

#Logs the phone number and phone level score between start_time and end_time.
# method is used to collapse frame level scores to phone level(sum or mean)
# end_time must be larger than start time
def log_phone_number_and_score(log_fh, labels, scores, start_time, end_time, method):
	    if end_time <= start_time :
	        raise Exception('End time: ' + str(end_time) + ' is not greater than start time: ' + str(start_time))
	    phone_number_start = get_phone_number_at_frame(labels, start_time)
	    phone_number_end = get_phone_number_at_frame(labels, end_time-1)
	    if phone_number_start != phone_number_end:
	        raise Exception('Phones at start and end time in labels differ')
	    phone_level_score = get_phone_score_from_frame_scores(scores, start_time, end_time, method)
	    log_fh.write( '[ ' + str(phone_number_start) + ' ' + str(phone_level_score) + ' ] ')


#This function takes a dataloader and writes the logids and paths of all samples
#in the dataloader to directory/filename. The generated sample list is used to train/test 
#the model for each fold. 
def generate_sample_list(dataloader, epa_root_path, directory, filename):
    sample_list_fh = open(directory + '/' + filename, 'w+')
    for i, data in enumerate(dataloader, 0):
        logids = unpack_logids_from_batch(data)
        for logid in logids:
            spkr_id = logid.split('_')[0]
            sample_path = epa_root_path + '/' + spkr_id + '/waveforms/' + logid + '.wav'
            sample_list_fh.write(logid + ' ' + sample_path + '\n')

#This function takes the path to an utterance list and returns a list of logids 
#and a dictionary mapping speaker ids to logid lists 
def generate_fileid_list_and_spkr2logid_dict(sample_list_path):
    file_id_list = []
    logids_by_speaker = {}
    sample_list_fh = open(sample_list_path, "r")
    for line in sample_list_fh.readlines():
        line = line.split()
        logid = line[0]
        speaker_id = logid.split('_')[0]
        sample_path = line[1]
        file_id_list.append(logid)
        if speaker_id in logids_by_speaker:
            logids_by_speaker[speaker_id].append(logid)
        else:
            logids_by_speaker[speaker_id] = [logid]
    return file_id_list, logids_by_speaker

#This function takes the path to an utterance list and returns a list of speaker ids
def get_speaker_list_from_utterance_list_file(sample_path):
    speaker_list = []
    sample_list_fh = open(sample_list_path, "r")
    for line in sample_list_fh.readlines():
        line = line.split()
        logid = line[0]
        speaker_id = logid.split('_')[0]
        if speaker_id not in speaker_list:
            speaker_list.append(speaker_id)
    return speaker_listr