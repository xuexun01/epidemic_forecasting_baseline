import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class CNNRNN_Res(nn.Module):
    def __init__(self, args, data): 
        super(CNNRNN_Res, self).__init__()
        self.ratio = 1.0   
        self.m = data.m  

        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=self.m, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True)
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=self.m, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN( input_size=self.m, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True)
        else:
            raise LookupError(' only support LSTM, GRU and RNN')

        self.residual_window = 4

        self.mask_mat = Parameter(torch.Tensor(self.m, self.m))
        nn.init.xavier_normal(self.mask_mat)  
        self.adj = data.adj  

        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(args.n_hidden, self.m)
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1)
        self.output = None
        output_fun = None
        if (output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (output_fun == 'tanh'):
            self.output = F.tanh


    def forward(self, x):
        masked_adj = self.adj * self.mask_mat
        x = x.matmul(masked_adj)
        r_out, _ = self.rnn(x)
        r = self.dropout(r_out[:,-1,:])
        res = self.linear1(r)
       
        if (self.residual_window > 0):
            z = x[:, -self.residual_window:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window)
            z = self.residual(z)
            z = z.view(-1,self.m)
            res = res * self.ratio + z

        if self.output is not None:
            res = self.output(res).float()
        return res