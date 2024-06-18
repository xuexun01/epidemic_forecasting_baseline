import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataloader import USState, JapanPrefecture
from torch.nn import Parameter
from torch.utils.data import DataLoader
from utils import *



class CNNRNN_Res(nn.Module):
    def __init__(self, hist_window, pred_window, n_nodes, n_hidden, n_layer, dropout, adj, output_fun=None): 
        super(CNNRNN_Res, self).__init__()
        self.ratio = 1.0
        self.n_nodes = n_nodes

        self.gru = nn.GRU(input_size=hist_window, hidden_size=n_hidden, num_layers=n_layer, dropout=dropout, batch_first=True)
        
        self.residual_window = 4

        self.mask_mat = Parameter(torch.Tensor(self.n_nodes, self.n_nodes))
        nn.init.xavier_normal_(self.mask_mat.data)
        self.adj = adj

        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(n_hidden, pred_window)
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, hist_window)
            self.residual = nn.Linear(self.residual_window, pred_window)
    

    def forward(self, x):
        # batch, hist_seq, node_num
        batch_size = x.shape[0]
        masked_adj = self.adj * self.mask_mat
        x = x.matmul(masked_adj)
        x = x.permute(0, 2, 1).contiguous()
        rnn_out, _ = self.gru(x)
        output = self.dropout(rnn_out)
        res = self.linear1(output)
       
        if (self.residual_window > 0):
            z = x[:, :, -self.residual_window:]
            z = z.permute(0, 2 ,1).contiguous().view(-1, self.residual_window)
            z = self.residual(z)
            z = z.view(batch_size, self.n_nodes, -1)
            res = res * self.ratio + z

        return res


arg = argparse.ArgumentParser()
arg.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
arg.add_argument('--n_hidden', type=int, default=32, help="rnn hidden states (could be set as any value)") 
arg.add_argument('--seed', type=int, default=3407, help='random seed')
arg.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
arg.add_argument('--lr', type=float, default=2e-3, help='initial learning rate')
arg.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 loss on parameters).')
arg.add_argument('--dropout', type=float, default=0.1, help='dropout rate usually 0.2-0.5.')
arg.add_argument('--batch_size', type=int, default=64, help="batch size")
arg.add_argument('--train_set_prop', type=float, default=.7, help="Training set proportion (0, 1)")
arg.add_argument('--vaild_set_prop', type=float, default=.1, help="Training set proportion (0, 1)")
arg.add_argument('--hist_len', type=int, default=14, help='')
arg.add_argument('--pred_len', type=int, default=8, help='leadtime default 5')
arg.add_argument('--save_dir', type=str,  default='./pt/cnnrnn_res.pt',help='dir path to save the final model')
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



model = CNNRNN_Res(hist_window=args.hist_len, pred_window=args.pred_len, n_nodes=n_nodes, n_hidden=args.n_hidden,
                   n_layer=args.n_layer, dropout=args.dropout, adj=adj).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)


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


test_model = CNNRNN_Res(hist_window=args.hist_len, pred_window=args.pred_len, n_nodes=n_nodes, n_hidden=args.n_hidden,
                   n_layer=args.n_layer, dropout=args.dropout, adj=adj).to(device)
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

mae /= len(test_loader)
mape /= len(test_loader)
rmse /= len(test_loader)
print(f"[Test loss] RMSE: {rmse} \t MAE: {mae} \t MAPE: {mape}")