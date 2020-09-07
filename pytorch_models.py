import torch
import torch.nn as nn
import torch.nn.functional as F

class FTDNNLayer(nn.Module):

    def __init__(self, semi_orth_in_dim, semi_orth_out_dim, out_dim, dropout_p=0.0):
        '''
        3 stage factorised TDNN http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
        '''
        super(FTDNNLayer, self).__init__()
        self.semi_orth_in_dim = semi_orth_in_dim
        self.semi_orth_out_dim = semi_orth_out_dim
        self.out_dim = out_dim
        self.dropout_p = dropout_p


        self.sorth = nn.Linear(self.semi_orth_in_dim, self.semi_orth_out_dim, bias=False)
        self.affine = nn.Linear(2*self.semi_orth_out_dim, self.out_dim, bias=True) 
        self.nl = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_dim, affine=False)
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        padding = x[:,0,:]
        xd = torch.cat([padding, x], axis=1)
        xd = xd[0,:-1,0]
        x = torch.cat([xd, x], axis=1)
        x = self.sorth(x)
        padding = x[:,-1,:]
        xd = torch.cat([x, padding], axis=1)
        xd = xd[0,1:,0]
        x = torch.cat([x, xd], axis=1)
        x = self.affine(x)
        x = self.nl(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class OutputXentLayer(nn.Module):

    def __init__(self, linear1_in_dim, linear2_in_dim, linear3_in_dim, out_dim, dropout_p=0.0):

        super(OutputXentLayer, self).__init__()
        self.linear1_in_dim = linear1_in_dim
        self.linear2_in_dim = linear2_in_dim
        self.linear3_in_dim = linear3_in_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(self.linear1_in_dim, self.linear2_in_dim, bias=True) 
        self.nl = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.linear2_in_dim, affine=False)
        self.linear2 = nn.Linear(self.linear2_in_dim, self.linear3_in_dim, bias=False) 
        self.bn2 = nn.BatchNorm1d(self.linear3_in_dim, affine=False)
        self.linear3 = nn.Linear(self.linear3_in_dim, self.out_dim, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.nl(x)
        x = self.bn1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.linear3(x)
        x = nn.Softmax(x)
        return x


class TDNN(nn.Module):

    def __init__(
        self,
        input_dim=220,
        output_dim=1536,
        batch_norm=True,
        dropout_p=0.0):

        super(TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.lda = nn.Linear(self.input_dim, self.input_dim)
        self.kernel = nn.Linear(self.input_dim,
                                self.output_dim)

        self.nonlinearity = nn.ReLU()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_dim, affine=False)
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(
            self.input_dim, d)

        mfccs = x[:,:,40]
        ivectors = x[:,:,-100:]
        padding_first = mfccs[:,0,:]
        padding_last = mfccs[:,-1,:]
        context_first = torch.cat([padding_first, mfccs[:,:-1,:]], axis=1)
        context_last = torch.cat([mfccs[:,1:,:], padding_last], axis=1)
        x = torch.cat([context_first, mfccs, context_last, ivectors], axis=1)
        x = self.lda(x)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        x = self.drop(x)

        if self.batch_norm:
            x = self.bn(x)
        return x.transpose(1, 2)


class FTDNN(nn.Module):

    def __init__(self, in_dim=220):

        super(FTDNN, self).__init__()

        self.layer01 = TDNN(input_dim=220, output_dim=1536)
        self.layer02 = FTDNNLayer(3072, 160, 1536)
        self.layer03 = FTDNNLayer(3072, 160, 1536)
        self.layer04 = FTDNNLayer(3072, 160, 1536)
        self.layer05 = FTDNNLayer(3072, 160, 1536)
        self.layer06 = FTDNNLayer(3072, 160, 1536)
        self.layer07 = FTDNNLayer(3072, 160, 1536)
        self.layer08 = FTDNNLayer(3072, 160, 1536)
        self.layer09 = FTDNNLayer(3072, 160, 1536)
        self.layer10 = FTDNNLayer(3072, 160, 1536)
        self.layer11 = FTDNNLayer(3072, 160, 1536)
        self.layer12 = FTDNNLayer(3072, 160, 1536)
        self.layer13 = FTDNNLayer(3072, 160, 1536)
        self.layer14 = FTDNNLayer(3072, 160, 1536)
        self.layer15 = FTDNNLayer(3072, 160, 1536)
        self.layer16 = FTDNNLayer(3072, 160, 1536)
        self.layer17 = FTDNNLayer(3072, 160, 1536)
        self.layer18 = nn.Linear(1536, 256, bias=False) #This is the prefinal-l layer
        self.layer19 = OutputXentLayer(256, 1536, 256, 6024)

    def sum_outputs_and_feed_to_layer(x, x_2, layer):
        x_3 = torch.sum(x*0.75, x_2)
        x = x_2
        x_2 = layer(x_3)
        return x, x_2

    def forward(self, x):

        '''
        Input must be (batch_size, seq_len, in_dim)
        '''
        x = self.layer01(x)
        x_2 = self.layer02(x)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer03)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer04)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer05)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer06)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer07)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer08)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer09)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer10)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer11)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer12)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer13)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer14)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer15)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer16)
        x, x_2 = sum_outputs_and_feed_to_layer(x,x_2, layer17)        
        x = layers18(x_2)
        x = layers19(x)



        return x

    def step_ftdnn_layers(self):
        for layer in self.children():
            if isinstance(layer, FTDNNLayer):
                layer.step_semi_orth()

    def set_dropout_alpha(self, alpha):
        for layer in self.children():
            if isinstance(layer, FTDNNLayer):
                layer.dropout.alpha = alpha

    def get_orth_errors(self):
        errors = 0.
        with torch.no_grad():
            for layer in self.children():
                if isinstance(layer, FTDNNLayer):
                    errors += layer.orth_error()
        return errors