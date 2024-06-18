import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
import scipy.sparse as sp
from utils import *
import argparse
import pandas as pd

from dataloader import USState, JapanPrefecture
from torch.utils.data import DataLoader

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 or len(sparse_mx.col)==0:
        print(sparse_mx.row,sparse_mx.col)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj2(adj):
    """Symmetrically normalize adjacency matrix."""
    # print(adj.shape)
    # adj += sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum+1e-5, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')' 



class cola_gnn(nn.Module):  
    def __init__(self, num_nodes, hist_len, pred_len, dropout, n_layer, n_hidden, k, hidsp, adj, origin_adj): 
        super().__init__()
        self.f_h = num_nodes
        self.num_nodes = num_nodes
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.adj = adj
        self.o_adj = origin_adj
        self.dropout = dropout
        self.n_hidden = n_hidden
        half_hid = int(self.n_hidden/2)

        self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(origin_adj.cpu().numpy())).to_dense().cuda()

        
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu
        self.Wb = Parameter(torch.Tensor(self.num_nodes, self.num_nodes))
        self.wb = Parameter(torch.Tensor(1))
        self.k = k
        self.conv = nn.Conv1d(1, self.k, self.hist_len)
        self.conv_long = nn.Conv1d(1, self.k, self.hist_len-self.k, dilation=2)
        self.n_spatial = hidsp

        self.conv1 = GraphConvLayer(self.k*9, self.n_hidden) # self.k
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_spatial)

        self.rnn = nn.LSTM( input_size=1, hidden_size=self.n_hidden, num_layers=n_layer, dropout=dropout, batch_first=True)

        # self.n_hidden = hidden_size BIDIRECTIONAL BUG
        self.out = nn.Linear(self.n_hidden + self.n_spatial, self.pred_len)  

        self.residual_window = 4
        self.ratio = 1.0
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, self.hist_len)
            self.residual = nn.Linear(self.residual_window, self.pred_len) 
        self.init_weights()
     

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)


    def forward(self, x, feat=None):
        '''
        Args:  x: (batch, time_step, m)  
            feat: [batch, window, dim, m]
        Returns: (batch, m)
        ''' 
        batch_size, hist_len, num_nodes = x.size()
        origin_x = x
        x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1)
        r_out, _ = self.rnn(x)
        last_hidden = r_out[:,-1,:]
        last_hidden = last_hidden.view(-1, self.num_nodes, self.n_hidden)
        out_temporal = last_hidden
        hid_rpt_m = last_hidden.repeat(1, self.num_nodes, 1).view(batch_size, self.num_nodes, self.num_nodes, self.n_hidden) # b,m,m,w continuous m
        hid_rpt_w = last_hidden.repeat(1, 1, self.num_nodes).view(batch_size, self.num_nodes, self.num_nodes, self.n_hidden) # b,m,m,w continuous w one window data
        a_mx = self.act( hid_rpt_m @ self.W1.t()  + hid_rpt_w @ self.W2.t() + self.b1 ) @ self.V + self.bv # row, all states influence one state 

        a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12, out=None)
        r_l = []
        r_long_l = []
        h_mids = origin_x

        for i in range(self.num_nodes):
            h_tmp = h_mids[:, :, i:i+1].permute(0,2,1).contiguous() 
            r = self.conv(h_tmp) # [batch, 10/k, 1]
            r_long = self.conv_long(h_tmp)
            r_l.append(r)
            r_long_l.append(r_long)

        r_l = torch.stack(r_l, dim=1)
        r_long_l = torch.stack(r_long_l,dim=1)
        r_l = torch.cat((r_l, r_long_l), -1)
        r_l = r_l.view(r_l.size(0), r_l.size(1), -1)
        r_l = torch.relu(r_l)

        adjs = self.adj.repeat(batch_size, 1)
        adjs = adjs.view(batch_size, self.num_nodes, self.num_nodes)
        c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        a_mx = adjs * c + a_mx * (1-c)

        adj = a_mx
        x = r_l
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        out_spatial = F.relu(self.conv2(x, adj))
        output = torch.cat((out_spatial, out_temporal),dim=-1)
        res = self.out(output)

        if (self.residual_window > 0):
            z = origin_x[:, -self.residual_window:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window)
            z = self.residual(z)
            z = z.view(batch_size, self.num_nodes, -1)
            res = res * self.ratio + z

        return res


arg = argparse.ArgumentParser()
arg.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
arg.add_argument('--n_hidden', type=int, default=32, help="rnn hidden states (could be set as any value)") 
arg.add_argument('--seed', type=int, default=3407, help='random seed')
arg.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
arg.add_argument('--lr', type=float, default=2e-3, help='initial learning rate')
arg.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 loss on parameters).')
arg.add_argument('--dropout', type=float, default=0.1, help='dropout rate usually 0.2-0.5.')
arg.add_argument('--batch_size', type=int, default=64, help="batch size")
arg.add_argument('--train_set_prop', type=float, default=.7, help="Training set proportion (0, 1)")
arg.add_argument('--vaild_set_prop', type=float, default=.1, help="Training set proportion (0, 1)")
arg.add_argument('--hist_len', type=int, default=14, help='')
arg.add_argument('--pred_len', type=int, default=14, help='leadtime default 5')
arg.add_argument('--save_dir', type=str,  default='./pt/colagnn.pt',help='dir path to save the final model')
arg.add_argument('--k', type=int, default=10,  help='kernels')
arg.add_argument('--hidsp', type=int, default=10,  help='spatial dim')
arg.add_argument('--dataset', type=str,  default='us')
args = arg.parse_args()


set_environment(args.seed)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


if args.dataset == "us":
    dates = pd.date_range(start="2020-04-12", end="2021-04-15")
    dates = list(dates.strftime("%Y_%m_%d"))

    train_dates = dates[:round(len(dates)*args.train_set_prop)]
    vaild_dates = dates[round(len(dates)*args.train_set_prop):round(len(dates)*(args.train_set_prop+args.vaild_set_prop))]
    test_dates = dates[round(len(dates)*(args.train_set_prop+args.vaild_set_prop)):]

    train_set = USState(args.hist_len, args.pred_len, data_path="./data/", dates=train_dates, device=device)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=False)
    valid_set = USState(args.hist_len, args.pred_len, data_path="./data/", dates=vaild_dates, device=device)
    valid_loader = DataLoader(valid_set, args.batch_size, shuffle=False)
    test_set = USState(args.hist_len, args.pred_len, data_path="./data/", dates=test_dates, device=device)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)
else:
    train_set = JapanPrefecture(args.hist_len, args.pred_len, data_path="./data/", device=device, train_ratio=args.train_set_prop, valid_ratio=args.vaild_set_prop, mode='train')
    train_loader = DataLoader(train_set, args.batch_size, shuffle=False)
    valid_set = JapanPrefecture(args.hist_len, args.pred_len, data_path="./data/", device=device, train_ratio=args.train_set_prop, valid_ratio=args.vaild_set_prop, mode='valid')
    valid_loader = DataLoader(valid_set, args.batch_size, shuffle=False)
    test_set = JapanPrefecture(args.hist_len, args.pred_len, data_path="./data/", device=device, train_ratio=args.train_set_prop, valid_ratio=args.vaild_set_prop, mode='test')
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)



origin_adj = train_set.origin_adj
adj = train_set.adj
inputs, _ = next(iter(train_loader))
n_nodes = inputs.shape[2]


model = cola_gnn(num_nodes=n_nodes, hist_len=args.hist_len, pred_len=args.pred_len, dropout=args.dropout, n_layer=args.n_layer, n_hidden=args.n_hidden,
                 k=args.k, hidsp=args.hidsp, origin_adj=origin_adj, adj=adj).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)


best_loss = np.Inf
for epoch in range(args.epochs):
    model.train()
    train_loss = 0.
    valid_loss = 0.
    model.zero_grad()
    optimizer.zero_grad()
    # train
    for inputs, labels in train_loader:
        labels = torch.transpose(labels, 1, 2)
        preds = model(inputs)
        loss_train = masked_mae(preds, labels)
        train_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    for inputs, labels in valid_loader:
        labels = torch.transpose(labels, 1, 2)
        preds = model(inputs)
        loss_valid = masked_mae(preds, labels)
        valid_loss += loss_valid.item()

    if valid_loss < best_loss:
        best_loss = valid_loss
        save_checkpoints(model, optimizer, epoch, args.save_dir)
    print("Epoch: {:03d} | Train Loss {:.3f} | Valid Loss {:.3f} | lr = {:.10f}".format(epoch+1, train_loss, valid_loss, optimizer.param_groups[0]['lr']))


test_model = cola_gnn(num_nodes=n_nodes, hist_len=args.hist_len, pred_len=args.pred_len, dropout=args.dropout, n_layer=args.n_layer, n_hidden=args.n_hidden,
                 k=args.k, hidsp=args.hidsp, origin_adj=origin_adj, adj=adj).to(device)
load_checkpoints(args.save_dir, test_model, optimizer)


test_model.eval()
mae, mape, rmse = 0., 0., 0.
for inputs, labels in test_loader:
    labels = torch.transpose(labels, 1, 2)
    preds = test_model(inputs)
    mae_loss, mape_loss, rmse_loss = metric(preds, labels)
    mae += mae_loss
    mape += mape_loss
    rmse += rmse_loss
print(f"[Test loss] RMSE: {rmse} \t MAE: {mae} \t MAPE: {mape}")