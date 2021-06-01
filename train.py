import os
import glob
from pathlib import Path

import torchaudio
import torch
import torch.optim as optim


from finetuning_utils import *
from utils import *
from dataset import *

from torch.utils.data import DataLoader, ConcatDataset

from pytorch_models import *

import wandb

from IPython import embed

from sklearn.model_selection import KFold



def criterion(batch_outputs, batch_labels):
    '''
    Calculates loss
    '''
    loss_fn = torch.nn.BCEWithLogitsLoss()
    batch_outputs, batch_labels = get_outputs_and_labels_for_loss(batch_outputs, batch_labels)
    #Calculate loss
    loss = loss_fn(batch_outputs, batch_labels)
    return loss

def train(model, trainloader, testloader, fold, run_name='test'):

    #Freeze all layers except the last
    for name, param in model.named_parameters():
        if 'layer19' not in name:
            param.requires_grad = False

    optimizer = optim.Adam(model.parameters())

    for epoch in range(10):  # loop over the dataset multiple times
        PATH = 'saved_state_dicts/' + run_name + '-fold-' + str(fold) + '-epoch-' + str(epoch) + '.pth'
        #If the checkpoint for the current epoch is already present, checkpoint is loaded and training is skipped
        if os.path.isfile(PATH):
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            continue

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):            
            #print("Batch " + str(i))
            # get the inputs; data is a list of (features, transcript, speaker_id, utterance_id, labels)
            inputs = unpack_features_from_batch(data)
            batch_labels = unpack_labels_from_batch(data)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            #print(outputs.size())

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            #print statistics
            running_loss += loss.item()

            if i % 20 == 19:    # log every 20 mini-batches
                print('Fold ' + str(fold), ' Epoch ' + str(epoch) + ' Batch ' + str(i))
                print('running_loss ' + str(running_loss/20))
                wandb.log({'train_loss_fold_' + str(fold): running_loss/20})
                running_loss = 0.0
                
        test_loss = test(model, testloader)
        wandb.log({'test_loss_fold_' + str(fold) : test_loss})
        
        torch.save(model.state_dict(), PATH)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)


    return model

def test(model, testloader):

    dataiter = iter(testloader)
    batch = dataiter.next()
    features = unpack_features_from_batch(batch)
    labels = unpack_labels_from_batch(batch)

    outputs = model(features)
    loss = criterion(outputs, labels)

    loss = loss.item()        

    return loss

def main():
    wandb.init(project="gop-finetuning")
    wandb.run.name = 'cross_validation_kaldi_phones'
    run_name = wandb.run.name

    dataset = EpaDB('EpaDB', 'epadb_full_path_list', 'phones_kaldi.txt', 'labels_with_kaldi_phones')

    seed = 42
    torch.manual_seed(seed)

    kfold = KFold(n_splits=5, shuffle=True, random_state = seed)

    spkr_list = dataset.get_speaker_list()

    for fold, (train_spkr_indexes, test_spkr_indexes) in enumerate(kfold.split(spkr_list)):

        train_sample_indexes = dataset.get_sample_indexes_from_spkr_indexes(train_spkr_indexes)
        test_sample_indexes  = dataset.get_sample_indexes_from_spkr_indexes(test_spkr_indexes)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_sample_indexes)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_sample_indexes)

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                     num_workers=1, sampler=train_subsampler, collate_fn=collate_fn_padd)

        testloader = torch.utils.data.DataLoader(dataset, batch_size=32, 
                                     num_workers=1, sampler=test_subsampler, collate_fn=collate_fn_padd)

        phone_count = dataset.phone_count()

        #Get acoustic model to train
        model = FTDNN(out_dim=phone_count)
        model.load_state_dict(torch.load('model_finetuning_kaldi.pt'))

        wandb.watch(model, log_freq=100)
        model = train(model, trainloader, testloader, fold, run_name=run_name)

main()