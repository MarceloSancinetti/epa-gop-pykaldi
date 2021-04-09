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
    Padds batch of variable length
    '''
    ## padd
    features = [ sample for sample, _,_,_,_ in batch ]
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    batch = [(features[i], batch[i][1], batch[i][2], batch[i][3], batch[i][4]) for i in range(len(features))]
    return batch


trainset = EpaDB('.')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                          shuffle=True, num_workers=1, collate_fn=collate_fn_padd)


#Por ahora uso el trainset para testear porque no se de donde sacar el dataset para testear
testloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=False, num_workers=2)


acoustic_model = FTDNN()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(acoustic_model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of (features, transcript, speaker_id, utterance_id, annotation)
        inputs = torch.stack([features for features, _,_,_,_ in data])
        annotations = torch.stack([annotation for _,_,_,_, annotation in data])


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = acoustic_model(inputs)

        #Aca hay que ver que onda el criterion
        loss = criterion(outputs, annotations)
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


