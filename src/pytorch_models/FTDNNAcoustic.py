import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from src.pytorch_models.FTDNN import FTDNN

class OutputLayer(nn.Module):

    def __init__(self, linear1_in_dim, linear2_in_dim, linear3_in_dim, out_dim):

        super(OutputLayer, self).__init__()
        self.linear1_in_dim = linear1_in_dim
        self.linear2_in_dim = linear2_in_dim
        self.linear3_in_dim = linear3_in_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(self.linear1_in_dim, self.linear2_in_dim, bias=True) 
        self.nl = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.linear2_in_dim, affine=False)
        self.linear2 = nn.Linear(self.linear2_in_dim, self.linear3_in_dim, bias=False) 
        self.bn2 = nn.BatchNorm1d(self.linear3_in_dim, affine=False)
        self.linear3 = nn.Linear(self.linear1_in_dim, self.out_dim, bias=True)
        #self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.nl(x)
        x = x.transpose(1,2)
        x = self.bn1(x).transpose(1,2)
        x = self.linear2(x)
        x = x.transpose(1,2)
        x = self.bn2(x).transpose(1,2)
        x = self.linear3(x)
        return x

class FTDNNAcoustic(nn.Module):

    def __init__(self):
        super(FTDNNAcoustic, self).__init__()
        self.ftdnn        = FTDNN()
        self.output_layer = OutputLayer(256, 1536, 256, 6024)

    def forward(self, x):

        '''
        Input must be (batch_size, seq_len, in_dim)
        '''
        x = self.ftdnn(x)
        x = self.output_layer(x)
        return x
