import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


'''STGCN'''
class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, hist_len, pred_len, n_hidden, origin_adj):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=n_hidden,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=n_hidden, out_channels=n_hidden,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=n_hidden, out_channels=n_hidden)
        self.fully = nn.Linear((hist_len - 2 * 5) * n_hidden, pred_len)
        
        self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(origin_adj.cpu().numpy())).to_dense().cuda()


    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        # batch_size, num_nodes, num_input_time_steps, num_features
        # [32, 20, 47]
        X = X.permute(0,2,1).contiguous()
        X = X.unsqueeze(-1)
        # print(X.shape)
        out1 = self.block1(X, self.adj)
        out2 = self.block2(out1, self.adj)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        # print(out4.shape)
        return out4.squeeze(-1)


arg = argparse.ArgumentParser()
arg.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
arg.add_argument('--n_hidden', type=int, default=16, help="rnn hidden states (could be set as any value)") 
arg.add_argument('--seed', type=int, default=3407, help='random seed')
arg.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
arg.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
arg.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 loss on parameters).')
arg.add_argument('--dropout', type=float, default=0.1, help='dropout rate usually 0.2-0.5.')
arg.add_argument('--batch_size', type=int, default=32, help="batch size")
arg.add_argument('--train_set_prop', type=float, default=.7, help="Training set proportion (0, 1)")
arg.add_argument('--vaild_set_prop', type=float, default=.1, help="Training set proportion (0, 1)")
arg.add_argument('--hist_len', type=int, default=14, help='')
arg.add_argument('--pred_len', type=int, default=3, help='leadtime default 5')
arg.add_argument('--save_dir', type=str,  default='./pt/stgcn.pt',help='dir path to save the final model')
arg.add_argument('--k', type=int, default=10,  help='kernels')
arg.add_argument('--hidsp', type=int, default=10,  help='spatial dim')
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

model = STGCN(num_nodes=n_nodes, num_features=1, hist_len=args.hist_len, pred_len=args.pred_len, n_hidden=args.n_hidden, origin_adj=origin_adj).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.6)


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
        loss_train = masked_rmse(preds, labels)
        train_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    for inputs, labels in valid_loader:
        labels = torch.transpose(labels, 1, 2)
        preds = model(inputs)
        loss_valid = masked_rmse(preds, labels)
        valid_loss += loss_valid.item()

    if valid_loss < best_loss:
        best_loss = valid_loss
        save_checkpoints(model, optimizer, epoch, args.save_dir)
    print("Epoch: {:03d} | Train Loss {:.3f} | Valid Loss {:.3f} | lr = {:.10f}".format(epoch+1, train_loss, valid_loss, optimizer.param_groups[0]['lr']))


test_model = STGCN(num_nodes=n_nodes, num_features=1, hist_len=args.hist_len, pred_len=args.pred_len, n_hidden=args.n_hidden, origin_adj=origin_adj).to(device)
load_checkpoints(args.save_dir, test_model, optimizer)

test_model.eval()
mae, mape, rmse = 0., 0., 0.
for inputs, labels in test_loader:
    labels = torch.transpose(labels, 1, 2)
    preds = test_model(inputs)
    mae_loss, mape_loss, rmse_loss = metric(preds, labels)
    print("[Test] RMSE: {} \t MAE: {} \t MAPE: {}".format(rmse_loss, mae_loss, mape_loss))
    mae += mae_loss
    mape += mape_loss
    rmse += rmse_loss

mae /= len(test_loader)
mape /= len(test_loader)
rmse /= len(test_loader)

print(f"[Test loss] RMSE: {rmse} \t MAE: {mae} \t MAPE: {mape}")