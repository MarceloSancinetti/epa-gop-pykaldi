import os
import glob
from pathlib import Path

import torchaudio
import torch
import torch.optim as optim

from utils import *
from dataset import *

from pytorch_models import *

import wandb

from IPython import embed


def unpack_logids_from_batch(batch):
    return [spkr_id + '_' + utt_id for _,_, spkr_id, utt_id,_,_ in batch]

def unpack_features_from_batch(batch):
    return torch.stack([features for features, _,_,_,_,_ in batch])

def unpack_labels_from_batch(batch):
    return torch.stack([labels for _,_,_,_,labels,_ in batch])

def unpack_annotations_from_batch(batch):
    return [annotation for _,_,_,_,_, annotation in batch]

def collate_fn_padd(batch):
    '''
    Padds batch of variable length (both features and labels)
    '''
    ## padd
    batch_features = [ features for features, _,_,_,_,_ in batch ]
    batch_features = torch.nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
    batch_labels = [ labels for _,_,_,_, labels,_ in batch ]
    batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True)
    batch = [(batch_features[i], batch[i][1], batch[i][2], batch[i][3], batch_labels[i], batch[i][5]) for i in range(len(batch))]
    return batch

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
    try: 
	    res = labels[frame].nonzero().item()
    except ValueError as e:
    	embed()
    return res

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