from pytorch_models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kaldiio import ReadHelper

model = FTDNN()
model.load_state_dict(torch.load('./state_dict.pt'))
model.eval()

ivector = {}
mfcc = 	{}
output = {}

with ReadHelper('ark:./epadb/test/data/raw_mfcc_test.1.ark') as reader:
    print("MFCCs")
    for key, numpy_array in reader:
        print(key)
        mfcc[key] = numpy_array

with ReadHelper('ark:./epadb/test/ivectors/ivector_online.2.ark') as reader:
    print("iVectors")
    for key, numpy_array in reader:
        print(key)
        ivector[key] = numpy_array

with ReadHelper('ark:./output.1.ark') as reader:
    print("Outputs")
    for key, numpy_array in reader:
        print(key)
        output[key] = numpy_array

x = np.repeat(ivector['spkr110_30'], 10, axis=0)


x = np.concatenate((x,mfcc['spkr110_30']), axis=1)
x = np.expand_dims(x, axis=0)

y = model(torch.from_numpy(x))
print(y)