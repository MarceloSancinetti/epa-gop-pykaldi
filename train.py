import os
import glob
from pathlib import Path

import torchaudio
import torch
import torch.optim as optim

from utils import *
from dataset import *

from pytorch_models import *

def collate_fn_padd(batch):
    '''
    Padds batch of variable length (both features and labels)
    '''
    ## padd
    batch_features = [ features for features, _,_,_,_ in batch ]
    batch_features = torch.nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
    batch_labels = [ labels for _,_,_,_, labels in batch ]
    batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True)
    batch = [(batch_features[i], batch[i][1], batch[i][2], batch[i][3], batch_labels[i]) for i in range(len(batch))]
    return batch


trainset = EpaDB('.', 'phones_epa.txt')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                          shuffle=True, num_workers=1, collate_fn=collate_fn_padd)


#Por ahora uso el trainset para testear porque no se de donde sacar el dataset para testear
testloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                          shuffle=False, num_workers=2)


phone_count = trainset.phone_count()
acoustic_model = FTDNN(out_dim=phone_count)


def criterion(outputs, batch_labels):
    print(outputs.shape)
    print(batch_labels.shape)
    print(batch_labels[0][100])
    print(outputs[0][100])
    samples_in_batch = batch_labels.shape[0]
    frame_count = batch_labels.shape[1]
    phone_count = batch_labels.shape[2]
    print(samples_in_batch)
    print(frame_count)
    print(phone_count)
    #for sample in 
    #for label in labels:
    #    for anot_tuple in annotation:
    #        correct_phone = 
    #return

optimizer = optim.SGD(acoustic_model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):            

        # get the inputs; data is a list of (features, transcript, speaker_id, utterance_id, labels)
        inputs = torch.stack([features for features, _,_,_,_ in data])
        batch_labels = torch.stack([labels for _,_,_,_, labels in data])


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = acoustic_model(inputs)

        #print(outputs.size())

        #Aca hay que ver que onda el criterion
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = './test.pth'
torch.save(acoustic_model.state_dict(), PATH)

dataiter = iter(testloader)
features, labels = dataiter.next()

outputs = acoustic_model(features)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


