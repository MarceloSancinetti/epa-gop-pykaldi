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
model.load_state_dict(torch.load('./state_dict.pt'))
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

with ReadHelper('ark:./output.5drop.ark') as reader:
    print("Outputs")
    for key, numpy_array in reader:
#        print(key)
        output[key] = numpy_array

#frames_per_chunk = 50
#online_ivector_extractor_period = 10


ivectors = ivector['spkr110_62']
ivectors_subsample = ivectors[2::5]
ivectors_subsample = np.repeat(ivectors_subsample, 5, axis=0)
ivectors[:-2] = ivectors_subsample
ivectors[-2:] = np.repeat(np.expand_dims(ivectors[-2], axis=0), 2, axis=0)

ivectors = np.repeat(ivectors, 10, axis=0)
#ivectors[-20:] = ivectors[-21]
#print(ivectors)
mfccs = mfcc['spkr110_62']

ivectors = ivectors[:mfccs.shape[0],:]

x = np.concatenate((mfccs,ivectors), axis=1)
x = np.expand_dims(x, axis=0)

y = model(torch.from_numpy(x))[0]
output = torch.from_numpy(output['spkr110_62'])



#y es el output de pytorch, output es el de kaldi
print(y)
print(y.shape)
print(output)
print(output.shape)

sanity = y/output
print(sanity)
for i in range (0,10):
    print(sanity[i])
