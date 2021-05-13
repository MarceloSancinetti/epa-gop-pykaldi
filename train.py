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

def train(model, trainloader, testloader, run_name='test'):

    #Freeze all layers except the last
    for name, param in model.named_parameters():
        if 'layer19' not in name:
            param.requires_grad = False

    optimizer = optim.Adam(model.parameters())

    for epoch in range(20):  # loop over the dataset multiple times

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

            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] train_loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                wandb.log({"train_loss": running_loss/20})
                running_loss = 0.0
                
        test_loss = test(model, testloader)
        wandb.log({"test_loss": test_loss})

    print('Finished Training')

    PATH = run_name + '.pth'
    torch.save(model.state_dict(), PATH)

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
    run_name = wandb.run.name

    trainset = EpaDB('.', 'epadb_train_path_list', 'phones_epa.txt')

    testset = EpaDB('.', 'epadb_test_path_list', 'phones_epa.txt')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                      shuffle=True, num_workers=2, collate_fn=collate_fn_padd)

    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                          shuffle=False, num_workers=2, collate_fn=collate_fn_padd)

    phone_count = trainset.phone_count()

    #Get acoustic model to train
    model = FTDNN(out_dim=phone_count)
    model.load_state_dict(torch.load('model_finetuning.pt'))

    wandb.watch(model, log_freq=100)
    model = train(model, trainloader, testloader, run_name=run_name)

main()