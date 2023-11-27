import argparse
import random
import numpy as np
import torch
from EpiGNN import EpiGNN
import torch.nn.functional as F
import pandas as pd
from dataload import MyDataset
from torch.utils.data import DataLoader

arg = argparse.ArgumentParser()
arg.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
arg.add_argument('--n_hidden', type=int, default=20, help="rnn hidden states (could be set as any value)") 
arg.add_argument('--seed', type=int, default=42, help='random seed')
arg.add_argument('--epochs', type=int, default=1500, help='number of epochs to train')
arg.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
arg.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
arg.add_argument('--dropout', type=float, default=0.2, help='dropout rate usually 0.2-0.5.')
arg.add_argument('--batch_size', type=int, default=64, help="batch size")
arg.add_argument('--check_point', type=int, default=1, help="check point")
arg.add_argument('--train', type=float, default=.7, help="Training ratio (0, 1)")
arg.add_argument('--hist_window', type=int, default=20, help='')
arg.add_argument('--pred_window', type=int, default=5, help='leadtime default 5')
arg.add_argument('--save_dir', type=str,  default='save',help='dir path to save the final model')
arg.add_argument('--lamda', type=float, default=0.01,  help='regularize params similarities of states')
arg.add_argument('--patience', type=int, default=100, help='patience default 100')
arg.add_argument('--n_kernel', type=int, default=8,  help='kernels')
arg.add_argument('--hidden_R', type=int, default=64,  help='hidden dim')
arg.add_argument('--hidden_A', type=int, default=64,  help='hidden dim of attention layer')
arg.add_argument('--hidden_P', type=int, default=1,  help='hidden dim of adaptive pooling')
arg.add_argument('--highway', type=int, default=1,  help='highway')
arg.add_argument('--extra', type=str, default='',  help='externel folder')
arg.add_argument('--n_layer_GCN', type=int, default=2, help='layer number of GCN')
arg.add_argument('--residual', type=int, default=0, help='0 means no residual link while 1 means need residual link')
arg.add_argument('--kernel_size', type=int, default=2, help='kernel size of temporal convolution network')
args = arg.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


dates = pd.date_range(start="2020-05-01", end="2021-12-30")
dates = list(dates.strftime("%Y_%m_%d"))


train_dates = dates[:round(len(dates)*args.train)]
test_dates = dates[round(len(dates)*args.train):]

train_set = MyDataset(args.hist_window, args.pred_window, data_path="./data/", dates=train_dates, device=device)
train_loader = DataLoader(train_set, args.batch_size, shuffle=False)

origin_adj = train_set.origin_adj
adj = train_set.adj
inputs, _ = next(iter(train_loader))
n_nodes = inputs.shape[2]

model = EpiGNN(n_nodes=n_nodes, hist_window=args.hist_window, pred_window=args.pred_window, n_layer=args.n_layer, dropout=args.dropout, hidden_R=args.hidden_R, hidden_A=args.hidden_R, hidden_P=args.hidden_P, 
               n_kernel=args.n_kernel, n_layer_GCN=args.n_layer_GCN, kernel_size=args.kernel_size, residual=args.residual, extra=args.extra, highway=args.highway, origin_adj=origin_adj, adj=adj).to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


for epoch in range(args.epochs):
    model.train()
    total_loss = 0.
    n_samples = 0.
    for inputs, preds in train_loader:
        optimizer.zero_grad()
        output,_  = model(inputs)
        if preds.size(0) == 1:
            preds = preds.view(-1)
        loss_train = F.mse_loss(output, preds) # mse_loss
        total_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
        n_samples += (output.size(0) * n_nodes)
    print("Epoch: {:03d} | Train Loss {:.3f} | lr = {:.10f}".format(epoch+1, total_loss/n_samples, args.lr))