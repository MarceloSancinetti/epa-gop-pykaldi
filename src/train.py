import os
import glob
from pathlib import Path
import argparse
import yaml


import torchaudio
import torch
import torch.optim as optim
import torch.multiprocessing as mp

from finetuning_utils import *
from utils import *
from dataset import *

from torch.utils.data import DataLoader, ConcatDataset

from pytorch_models import *

import wandb

from IPython import embed

from sklearn.model_selection import KFold



def get_model_path_for_fold(model_path, fold, layer_amount):
    #This is used to allow training to start from a previous experiment's
    #state_dict with the same fold
    return model_path.replace("@FOLD@", str(fold)) 

def get_path_for_checkpoint(state_dict_dir, run_name, fold, epoch):
    return state_dict_dir + run_name + '-fold-' + str(fold) + '-epoch-' + str(epoch) + '.pth'

def start_from_checkpoint(PATH, model, optimizer):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return model, optimizer, step        

def freeze_layers_for_finetuning(model, layer_amount):
    #Generate layer names for layers that should be trained
    layers_to_train = ['layer' + str(19 - x) for x in range(layer_amount)]

    #Freeze all layers except #layer_amount layers starting from the last
    for name, module in model.named_modules():
        freeze_layer = all([layer not in name for layer in layers_to_train])
        if freeze_layer:
            module.eval()

#This function calculates the loss for a specific phone in the phone set given the outputs and labels
def loss_for_phone(outputs, labels, phone_count, phone, phone_weights):
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        phone_mask = torch.zeros(phone_count)
        phone_mask[phone] = 1
        labels_for_phone = labels * phone_mask 
        outputs, labels = get_outputs_and_labels_for_loss(outputs, labels_for_phone)
        #Calculate loss
        occurrences = labels.shape[0]
        phone_weight = phone_weights['phone' + str(phone)]
        embed()
        if occurrences <= 1:
            return 0
        else:
            return loss_fn(outputs, labels) * phone_weight

#Returns total batch loss, adding the loss computed for each class individually
def criterion(batch_outputs, batch_pos_labels, batch_neg_labels, phone_count, phone_weights):
    '''
    Calculates loss
    '''
    loss = 0
    for phone in range(phone_count):
        loss += loss_for_phone(batch_outputs, batch_pos_labels, phone_count, phone, phone_weights)

    for phone in range(phone_count):
        loss += loss_for_phone(batch_outputs, batch_neg_labels, phone_count, phone, phone_weights)
    
    return loss

def train(model, trainloader, testloader, phone_count, phone_weights, fold, epochs, state_dict_dir, run_name, layer_amount, lr, use_clipping):
    print("Started training fold " + str(fold))

    step = 0

    freeze_layers_for_finetuning(model, layer_amount)

    optimizer = optim.Adam(model.parameters(), lr=lr)#, weight_decay=1e-5)

    for epoch in range(epochs):  # loop over the dataset multiple times
        PATH = get_path_for_checkpoint(state_dict_dir, run_name, fold, epoch) 
        #If the checkpoint for the current epoch is already present, checkpoint is loaded and training is skipped
        if os.path.isfile(PATH):
            model, optimizer, step = start_from_checkpoint(PATH, model, optimizer)
            continue

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):            
            #print("Batch " + str(i))
            # get the inputs; data is a list of (features, transcript, speaker_id, utterance_id, labels)
            logids = unpack_logids_from_batch(data)
            inputs = unpack_features_from_batch(data)
            batch_pos_labels = unpack_pos_labels_from_batch(data)
            batch_neg_labels = unpack_neg_labels_from_batch(data)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, batch_pos_labels, batch_neg_labels, phone_count, phone_weights)
            
            if epoch == 0 and i == 0:
               wandb.log({'train_loss_fold_' + str(fold): loss,
                          'step' : step})

            loss.backward()
            if use_clipping=='true':
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True, norm_type=2)
            optimizer.step()

            if loss > 1000000:
                print(logids)
#                embed()


            #print statistics
            running_loss += loss.item()

            if i % 5 == 4:    # log every 20 mini-batches
                print('Fold ' + str(fold), ' Epoch ' + str(epoch) + ' Batch ' + str(i))
                print('running_loss ' + str(running_loss/5))
                wandb.log({'train_loss_fold_' + str(fold): running_loss/5,
                           'step' : step,
                           'epoch': epoch})
                step += 1
                running_loss = 0.0
                
        test_loss = test(model, testloader, phone_count)
        wandb.log({'test_loss_fold_' + str(fold) : test_loss,
                   'step' : step})
        step += 1
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step
            }, PATH)


    return model

def test(model, testloader, phone_count):

    dataiter = iter(testloader)
    batch = dataiter.next()
    features = unpack_features_from_batch(batch)
    pos_labels = unpack_pos_labels_from_batch(batch)
    neg_labels = unpack_neg_labels_from_batch(batch)


    outputs = model(features)
    loss = criterion(outputs, pos_labels, neg_labels, phone_count, phone_weights)

    loss = loss.item()        

    return loss

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', dest='run_name', help='Run name', default=None)
    parser.add_argument('--utterance-list', dest='utterance_list', help='File with utt list', default=None)
    parser.add_argument('--folds', dest='fold_amount', help='Amount of folds to use in training', default=None)
    parser.add_argument('--epochs', dest='epoch_amount', help='Amount of epochs to use in training', default=None)
    parser.add_argument('--layers', dest='layer_amount', help='Amount of layers to train starting from the last (if layers=1 train only the last layer)', default=None)
    parser.add_argument('--learning-rate', dest='learning_rate', help='Learning rate to use during training', type=float, default=None)
    parser.add_argument('--batch-size', dest='batch_size', help='Batch size for training', type=int, default=None)
    parser.add_argument('--use-clipping', dest='use_clipping', help='Whether to use gradien clipping or not', default=None)
    parser.add_argument('--phones-file', dest='phones_file', help='File with list of phones', default=None)
    parser.add_argument('--labels-dir', dest='labels_dir', help='Directory with labels used in training', default=None)
    parser.add_argument('--model-path', dest='model_path', help='Path to .pth/pt file with model to finetune', default=None)
    parser.add_argument('--phone-weights-path', dest='phone_weights_path', help='Path to .yaml containing weights for phone-level loss', default=None)
    parser.add_argument('--epa-root-path', dest='epa_root_path', help='EpaDB root path', default=None)
    parser.add_argument('--features-path', dest='features_path', help='Path to features directory', default=None)
    parser.add_argument('--conf-path', dest='conf_path', help='Path to config directory used in feature extraction', default=None)
    parser.add_argument('--test-sample-list-dir', dest='test_sample_list_dir', help='Path to output directory to save test sample lists', default=None)
    parser.add_argument('--state-dict-dir', dest='state_dict_dir', help='Path to output directory to save state dicts', default=None)
    parser.add_argument('--use-multi-process', dest='use_multi_process', help='Whether to use multiple processes or not', default=None)

    args         = parser.parse_args()
    run_name     = args.run_name
    folds        = int(args.fold_amount)
    epochs       = int(args.epoch_amount)
    layer_amount = int(args.layer_amount)

    wandb.init(project="gop-finetuning")
    wandb.run.name = run_name

    epa_root_path = args.epa_root_path
    dataset = EpaDB(epa_root_path, args.utterance_list, args.phones_file, args.labels_dir, args.features_path, args.conf_path)

    seed = 42
    torch.manual_seed(seed)

    kfold = KFold(n_splits=folds, shuffle=True, random_state = seed)

    spkr_list = dataset.get_speaker_list()

    phone_weights_fh = open(args.phone_weights_path)
    phone_weights = yaml.safe_load(phone_weights_fh)

    for fold, (train_spkr_indexes, test_spkr_indexes) in enumerate(kfold.split(spkr_list)):


        train_sample_indexes = dataset.get_sample_indexes_from_spkr_indexes(train_spkr_indexes)
        test_sample_indexes  = dataset.get_sample_indexes_from_spkr_indexes(test_spkr_indexes)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_sample_indexes)
        test_subsampler  = torch.utils.data.SubsetRandomSampler(test_sample_indexes)

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                     num_workers=1, sampler=train_subsampler, collate_fn=collate_fn_padd)

        testloader = torch.utils.data.DataLoader(dataset, batch_size=32, 
                                     num_workers=1, sampler=test_subsampler, collate_fn=collate_fn_padd)

        phone_count = dataset.phone_count()

        #Get acoustic model to train
        model = FTDNN(out_dim=phone_count) 
        state_dict = torch.load(get_model_path_for_fold(args.model_path, fold, layer_amount))
        model.load_state_dict(state_dict['model_state_dict'])

        #Train the model
        wandb.watch(model, log_freq=100)
        if args.use_multi_process == "true":
            processes = []
            p = mp.Process(target=train, args=(model, trainloader, testloader, phone_count, phone_weights, fold, 
                           epochs, args.state_dict_dir, run_name, layer_amount, args.learning_rage, args.use_clipping))
            p.start()
            processes.append(p)
        else:
            train(model, trainloader, testloader, phone_count, phone_weights, 
                  fold, epochs, args.state_dict_dir, run_name, layer_amount, args.learning_rate, args.use_clipping)

        #Generate test sample list for current fold
        generate_test_sample_list(testloader, epa_root_path, args.test_sample_list_dir, 'test_sample_list_fold_' + str(fold))

    if args.use_multi_process == "true":
        for p in processes:
            p.join()

if __name__ == '__main__':
    main()
