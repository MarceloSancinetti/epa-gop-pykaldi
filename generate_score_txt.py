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


def generate_score_txt(model, testloader, score_file_name, phone_dict):

    print('Writing scores to .txt')
    for i, batch in enumerate(testloader, 0):
        score_log_fh = open(score_file_name, "a+")
        print('Batch ' + str(i+1) + '/' + str(len(testloader)))

        logids = unpack_logids_from_batch(batch)
        features = unpack_features_from_batch(batch)
        labels = unpack_labels_from_batch(batch)
        annotations = unpack_annotations_from_batch(batch)

        outputs = (-1) * model(features)

        frame_level_scores = get_scores_for_canonic_phones(outputs, labels)
        #Iterate over samples in the test batch
        for i, logid in enumerate(logids):
            score_log_fh.write(logid + ' ')
            #Iterate over phones in the annotation for the current sample
            for phone_name, start_time, end_time in annotations[i]:
                #Check if the phone was pronounced
                if start_time != end_time:
                    #Log the score for the current frame in the annotation
                    log_phone_number_and_score(score_log_fh, labels[i], 
                    frame_level_scores[i], start_time, end_time, 'mean')
                else:
                    try:
                        phone_number = phone_dict[phone_name]
                    except KeyError as e:
                        embed()
                    score_log_fh.write( '[ ' + str(phone_number) + ' -1000'  + ' ] ')
            score_log_fh.write('\n')
    
        score_log_fh.close()           


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict-dir', dest='state_dict_dir', help='Directory to saved state dicts in .pth', default=None)
    parser.add_argument('--model-name', dest='model_name', help='Model name (usually the name of the wandb run that generated the .pth file)', default=None)
    parser.add_argument('--gop-txt-dir', dest='gop_txt_dir', help='Directory to save generated scores', default=None)
    args = parser.parse_args()

    state_dict_dir = args.state_dict_dir
    model_name = args.model_name
    gop_txt_dir = args.gop_txt_dir

    testset = EpaDB('.', 'epadb_test_path_list', 'phones_epa.txt')
    testloader = torch.utils.data.DataLoader(testset, batch_size=2,
                                          shuffle=False, num_workers=1, collate_fn=collate_fn_padd)

    phone_count = testset.phone_count()

    #Get acoustic model to test
    model = FTDNN(out_dim=phone_count)
    model.load_state_dict(torch.load(state_dict_dir + '/' + model_name + '.pth'))

    phone_dict = testset._pure_phone_dict

    generate_score_txt(model, testloader, gop_txt_dir+ '/' +'gop-'+model_name+'.txt', phone_dict)


main()