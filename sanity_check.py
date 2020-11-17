from pytorch_models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kaldiio import ReadHelper
import math

def rms(vector):
    return np.sqrt(np.sum(np.square(vector)))

def batchnorm(matrix, mean, var):
    #mean = np.mean(matrix, axis=0)
    #var = np.var(matrix, axis=0)
    res = (matrix - mean)/np.sqrt(var + 0.001)
    print(var)
    return res


model = FTDNN()
model.load_state_dict(torch.load('./model.pt'))
model.eval()

ivector = {}
mfcc = 	{}
output = {}

with ReadHelper('ark:./features_one/raw_mfcc_test.1.ark') as reader:
    print("MFCCs")
    for key, numpy_array in reader:
        print(key)
        mfcc[key] = numpy_array

with ReadHelper('ark:./features_one/ivector_online.1.ark') as reader:
    print("iVectors")
    for key, numpy_array in reader:
        print(key)
        ivector[key] = numpy_array

with ReadHelper('ark:./output.1drop.ark') as reader:
    print("Outputs")
    for key, numpy_array in reader:
        print(key)
        output[key] = numpy_array



#frames_per_chunk = 50
#online_ivector_extractor_period = 10


ivectors = ivector['spkr110_62']
ivectors_subsample = ivectors[2::5]
ivectors_subsample = np.repeat(ivectors_subsample, 5, axis=0)
ivectors[:-2] = ivectors_subsample
ivectors[-2:] = np.repeat(np.expand_dims(ivectors[-2], axis=0), 2, axis=0)

ivectors = np.repeat(ivectors, 10, axis=0)
print(ivectors.shape)
#ivectors[-20:] = ivectors[-21]
#print(ivectors)
mfccs = mfcc['spkr110_62']
print(mfccs.shape)

ivectors = ivectors[:mfccs.shape[0],:]

x = np.concatenate((mfccs,ivectors), axis=1)
x = np.expand_dims(x, axis=0)

y = model(torch.from_numpy(x))[0]
output = torch.from_numpy(output['spkr110_62'])

state_dict = model.state_dict()
lp = state_dict['layer02.sorth.weight']


#y es el output de pytorch, output es el de kaldi
print(y)
print(y.shape)
print(output)
print(output.shape)

#sanity = y/output
#print(sanity)
#for i in range (30,90):
#    print(i, sanity[i])
#print(sanity[80])

input = {}
params = {}

with ReadHelper('ark:conv_step/input-0.544782 nopad.ark') as reader:
    print("Input")
    i = 0
    for key, numpy_array in reader:
        print(key)
        input[str(i)] = numpy_array
        i = i+1

with ReadHelper('ark:conv_step/params-0.544782.ark') as reader:
    print("Params")
    i = 0
    for key, numpy_array in reader:
        print(key)
        params[str(i)] = numpy_array
        i = i+1
print("Input mio")
print(y[0:50,:1536])
#print(lp)
print(y[0:50,:1536].shape)
print("Input Kaldi")
print(input['0'][0:50,:1536])
print(input['0'][0:50,:1536].shape)
for i in range(1,50):
    equal = True
    for j in range(0, 100):
        if input['0'][0,j] != input['0'][i,j]:
            equal = False
    if equal:
        print("Input: ", i)