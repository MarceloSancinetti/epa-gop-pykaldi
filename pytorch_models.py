import torch
import torch.nn as nn
import torch.nn.functional as F

class FTDNNLayer(nn.Module):

    def __init__(self, semi_orth_in_dim affine_in_dim, out_dim, dropout_p=0.0):
        '''
        3 stage factorised TDNN http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
        '''
        super(FTDNNLayer, self).__init__()


        self.sorth = nn.Linear(self.semi_orth_in_dim, self.affine_in_dim, bias=False)
        self.affine = nn.Linear(self.affine_in_dim, self.out_dim, bias=True) 
        self.nl = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = Dropout(p=self.dropout_p)

    def forward(self, x):
        padding = x[:,0,:]
        xd = torch.cat((padding, x), axis=1)
        xd = xd[0,:-1,0]
        x = torch.cat((xd, x), axis=1)
        x = self.sorth(x)
        padding = x[:,-1,:]
        xd = torch.cat((x, padding), axis=1)
        xd = xd[0,1:,0]
        x = torch.cat((x, xd), axis=1)
        x = self.affine(x)
        x = self.nl(x)
        x = self.bn(x)
        x = self.dropout(x)
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

        self.lda = Linear(self.input_dim, self.input_dim)
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
        context_first = torch.cat((padding_first, mfccs[:,:-1,:]), axis=1)
        context_last = torch.cat((mfccs[:,1:,:], padding_last), axis=1)
        x = torch.cat((context_first, mfccs, context_last, ivectors), axis=1)
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


        #Todo esto ya no hace falta, hay que agregar las prefinal
        #self.layer10 = DenseReLU(1024, 2048)
        #self.layer11 = StatsPool()

        #self.layer12 = DenseReLU(4096, 512)

    def forward(self, x):

        #Este es el forward viejo que ya no tiene sentido, hay que ver cómo se empastan las layers
        '''
        Input must be (batch_size, seq_len, in_dim)
        '''
        x = self.layer01(x)
        x_2 = self.layer02(x)
        x_3 = self.layer03(x_2)
        x_4 = self.layer04(x_3)
        skip_5 = torch.cat([x_4, x_3], dim=-1)
        x = self.layer05(skip_5)
        x_6 = self.layer06(x)
        skip_7 = torch.cat([x_6, x_4, x_2], dim=-1)
        x = self.layer07(skip_7)
        x_8 = self.layer08(x)
        skip_9 = torch.cat([x_8, x_6, x_4], dim=-1)
        x = self.layer09(skip_9)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
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