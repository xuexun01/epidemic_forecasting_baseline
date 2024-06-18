import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

import argparse
import random
import numpy as np
import pandas as pd
from dataloader import USState, JapanPrefecture
from torch.utils.data import DataLoader
from utils import *


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ELU()
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
            return self.act(output + self.bias)
        else:
            return self.act(output)

class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        # embedding [batchsize, hidden_dim]
        nodevec1 = self.linear1(embedding)
        nodevec2 = self.linear2(embedding)
        nodevec1 = self.alpha * nodevec1
        nodevec2 = self.alpha * nodevec2
        nodevec1 = torch.tanh(nodevec1)
        nodevec2 = torch.tanh(nodevec2)
        
        adj = torch.bmm(nodevec1, nodevec2.permute(0, 2, 1))-torch.bmm(nodevec2, nodevec1.permute(0, 2, 1))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj

class ConvBranch(nn.Module):
    def __init__(self, m, in_channels, out_channels, kernel_size, dilation_factor, hidP=1, isPool=True):
        super().__init__()
        self.m = m
        self.isPool = isPool
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size,1), dilation=(dilation_factor,1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        if self.isPool:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        #self.activate = nn.Tanh()
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool:
            x = self.pooling(x)
        x = x.view(batch_size, -1, self.m)
        return x

class RegionAwareConv(nn.Module):
    def __init__(self, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        self.P = P
        self.m = m
        self.k = k
        self.hidP = hidP
        self.conv_l1 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=3, dilation_factor=1, hidP=self.hidP)
        self.conv_l2 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=5, dilation_factor=1, hidP=self.hidP)
        self.conv_p1 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=3, dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_p2 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=5, dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_g = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=self.P, dilation_factor=1, hidP=None, isPool=False)
        self.activate = nn.Tanh()
    
    def forward(self, x):
        x = x.view(-1, 1, self.P, self.m)
        # local pattern
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)
        # periodic pattern
        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)
        # global
        x_global = self.conv_g(x)
        # concat and activate
        x = torch.cat([x_local, x_period, x_global], dim=1).permute(0, 2, 1)
        x = self.activate(x)
        return x


def getLaplaceMat(batch_size, m, adj):
    i_mat = torch.eye(m).to(adj.device)
    i_mat = i_mat.unsqueeze(0)
    o_mat = torch.ones(m).to(adj.device)
    o_mat = o_mat.unsqueeze(0)
    i_mat = i_mat.expand(batch_size, m, m)
    o_mat = o_mat.expand(batch_size, m, m)
    adj = torch.where(adj>0, o_mat, adj)
    d_mat = torch.sum(adj, dim=2) # attention: dim=2
    d_mat = d_mat.unsqueeze(2)
    d_mat = d_mat + 1e-12
    d_mat = torch.pow(d_mat, -0.5)
    d_mat = d_mat.expand(d_mat.shape[0], d_mat.shape[1], d_mat.shape[1])
    d_mat = i_mat * d_mat

    # laplace_mat = d_mat * adj * d_mat
    laplace_mat = torch.bmm(d_mat, adj)
    laplace_mat = torch.bmm(laplace_mat, d_mat)
    return laplace_mat


class EpiGNN(nn.Module):
    def __init__(self, n_nodes, hist_window, pred_window, n_layer, dropout, hidden_R, hidden_A, hidden_P, n_kernel, n_layer_GCN, kernel_size, residual, highway, origin_adj, adj):
        super().__init__()
        self.n_nodes = n_nodes
        self.hist_window = hist_window
        self.n_layer = n_layer
        self.hidR = hidden_R
        self.hidA = hidden_A
        self.hidP = hidden_P
        self.n_kernel = n_kernel
        self.kernel_size = kernel_size
        self.n_layer_GCN = n_layer_GCN
        self.residual = residual
        self.hw = highway
        self.dropout = nn.Dropout(dropout)
        self.adj = adj

        if self.hw > 0:
            self.highway = nn.Linear(self.hw, pred_window)

        # Feature embedding
        self.hidR = self.n_kernel*4*self.hidP + self.n_kernel
        self.backbone = RegionAwareConv(P=self.hist_window, m=self.n_nodes, k=self.n_kernel, hidP=self.hidP)

        # global
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.t_enc = nn.Linear(1, self.hidR)

        # local
        self.degree = torch.sum(origin_adj, dim=-1)
        self.s_enc = nn.Linear(1, self.hidR)

        # external resources
        self.external_parameter = nn.Parameter(torch.FloatTensor(self.n_nodes, self.n_nodes), requires_grad=True)

        # Graph Generator and GCN
        self.d_gate = nn.Parameter(torch.FloatTensor(self.n_nodes, self.n_nodes), requires_grad=True)
        self.graphGen = GraphLearner(self.hidR)
        self.GNNBlocks = nn.ModuleList([GraphConvLayer(in_features=self.hidR, out_features=self.hidR) for i in range(self.n_layer_GCN)])

        # prediction
        if self.residual == 0:
            self.output = nn.Linear(self.hidR*2, pred_window)
        else:
            self.output = nn.Linear(self.hidR*(self.n_layer_GCN+1), pred_window)
        self.init_weights()
     

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
    

    def forward(self, inputs, isEval=False):
        batch_size = inputs.shape[0] # batchsize, w, m

        # step 1: Use multi-scale convolution to extract feature embedding (SEFNet => RAConv).
        temp_emb = self.backbone(inputs)

        # step 2: generate global transmission risk encoding.
        query = self.WQ(temp_emb) # batch, N, hidden
        query = self.dropout(query)
        key = self.WK(temp_emb)
        key = self.dropout(key)
        attn = torch.bmm(query, key.transpose(1, 2))
        #attn = self.leakyrelu(attn)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1)
        attn = attn.unsqueeze(2)
        t_enc = self.t_enc(attn)
        t_enc = self.dropout(t_enc)

        # step 3: generate local transmission risk encoding.
        # print(self.degree.shape) [self.m]
        d = self.degree.unsqueeze(1)
        s_enc = self.s_enc(d)
        s_enc = self.dropout(s_enc)

        # Three embedding fusion.
        feat_emb = temp_emb + t_enc + s_enc

        # step 4: Region-Aware Graph Learner
        # apply Graph Learner to generate a graph
        d_mat = torch.mm(d, d.permute(1, 0))
        d_mat = torch.mul(self.d_gate, d_mat)
        d_mat = torch.sigmoid(d_mat)
        spatial_adj = torch.mul(d_mat, self.adj)
        adj = self.graphGen(temp_emb)
        
        # if additional information => fusion
        adj = adj + spatial_adj

        # get laplace adjacent matrix
        laplace_adj = getLaplaceMat(batch_size, self.n_nodes, adj)
        
        # Graph Convolution Network
        node_state = feat_emb
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = layer(node_state, laplace_adj)
            node_state = self.dropout(node_state)
            node_state_list.append(node_state)

        # Final prediction
        node_state = torch.cat([node_state, feat_emb], dim=-1)
        res = self.output(node_state).squeeze(2)
        # highway means autoregressive model
        if self.hw > 0:
            z = inputs[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(batch_size, self.n_nodes, -1)
            res = res + z
        
        # if evaluation, return some intermediate results
        if isEval:
            imd = (adj, attn)
        else:
            imd = None
        return res, imd



arg = argparse.ArgumentParser()
arg.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
arg.add_argument('--n_hidden', type=int, default=32, help="rnn hidden states (could be set as any value)")
arg.add_argument('--seed', type=int, default=3407, help='random seed')
arg.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
arg.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
arg.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 loss on parameters).')
arg.add_argument('--dropout', type=float, default=0.1, help='dropout rate usually 0.2-0.5.')
arg.add_argument('--batch_size', type=int, default=64, help="batch size")
arg.add_argument('--train_set_prop', type=float, default=.7, help="Training set proportion (0, 1)")
arg.add_argument('--vaild_set_prop', type=float, default=.1, help="Training set proportion (0, 1)")
arg.add_argument('--hist_len', type=int, default=14, help='')
arg.add_argument('--pred_len', type=int, default=14, help='leadtime default 5')
arg.add_argument('--save_dir', type=str,  default='./pt/epignn.pt',help='dir path to save the final model')
arg.add_argument('--lamda', type=float, default=0.01,  help='regularize params similarities of states')
arg.add_argument('--patience', type=int, default=100, help='patience default 100')
arg.add_argument('--n_kernel', type=int, default=10,  help='kernels')
arg.add_argument('--hidden_R', type=int, default=32,  help='hidden dim')
arg.add_argument('--hidden_A', type=int, default=32,  help='hidden dim of attention layer')
arg.add_argument('--hidden_P', type=int, default=1,  help='hidden dim of adaptive pooling')
arg.add_argument('--highway', type=int, default=1,  help='highway')
arg.add_argument('--n_layer_GCN', type=int, default=1, help='layer number of GCN')
arg.add_argument('--residual', type=int, default=0, help='0 means no residual link while 1 means need residual link')
arg.add_argument('--kernel_size', type=int, default=2, help='kernel size of temporal convolution network')
arg.add_argument('--dataset', type=str,  default='japan')
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

model = EpiGNN(n_nodes=n_nodes, hist_window=args.hist_len, pred_window=args.pred_len, n_layer=args.n_layer, dropout=args.dropout, hidden_R=args.hidden_R, hidden_A=args.hidden_R, hidden_P=args.hidden_P, 
               n_kernel=args.n_kernel, n_layer_GCN=args.n_layer_GCN, kernel_size=args.kernel_size, residual=args.residual, highway=args.highway, origin_adj=origin_adj, adj=adj).to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)


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
        preds, _ = model(inputs)
        loss_train = masked_mae(preds, labels)
        train_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    for inputs, labels in valid_loader:
        labels = torch.transpose(labels, 1, 2)
        preds, _ = model(inputs)
        loss_valid = masked_mae(preds, labels)
        valid_loss += loss_valid.item()

    if valid_loss < best_loss:
        best_loss = valid_loss
        save_checkpoints(model, optimizer, epoch, args.save_dir)
    print("Epoch: {:03d} | Train Loss {:.3f} | Valid Loss {:.3f} | lr = {:.10f}".format(epoch+1, train_loss, valid_loss, optimizer.param_groups[0]['lr']))


# test
test_model = EpiGNN(n_nodes=n_nodes, hist_window=args.hist_len, pred_window=args.pred_len, n_layer=args.n_layer, dropout=args.dropout, hidden_R=args.hidden_R, hidden_A=args.hidden_R, hidden_P=args.hidden_P, 
               n_kernel=args.n_kernel, n_layer_GCN=args.n_layer_GCN, kernel_size=args.kernel_size, residual=args.residual, highway=args.highway, origin_adj=origin_adj, adj=adj).to(device)
load_checkpoints(args.save_dir, test_model, optimizer)


test_model.eval()
mae, mape, rmse = 0., 0., 0.
for inputs, labels in test_loader:
    labels = torch.transpose(labels, 1, 2)
    preds, _ = test_model(inputs)
    mae_loss, mape_loss, rmse_loss = metric(preds, labels)
    mae += mae_loss
    mape += mape_loss
    rmse += rmse_loss

mae /= len(test_loader)
mape /= len(test_loader)
rmse /= len(test_loader)

print(f"[Test loss] RMSE: {rmse} \t MAE: {mae} \t MAPE: {mape}")