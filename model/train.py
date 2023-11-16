import argparse

import numpy as np
import pandas as pd
import torch.nn as nn
from dataloader import CCDataset
from torch.utils.data import DataLoader
from utils import *
from LSTM import LSTM, LSTM_attn
from sklearn.preprocessing import MinMaxScaler


parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--learn_rate", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=0)
args = parser.parse_args()

# initialize
logger = config_logger(logfile="./output/run.log")
logger.info("successfully configure the logger...")
logger.info("initialize the environment...")
set_environment(seed=144273)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = read_properties("./config.properties")

dates = pd.date_range(start="2021-01-01", end="2021-03-30")
dates = list(dates.strftime("%Y_%m_%d"))

mydataset = CCDataset(hist_window=6, pred_window=1, data_path="./data/", dates=dates, device="cuda:0")
myloader = DataLoader(mydataset, batch_size=10, shuffle=False)

model = LSTM(input_dim=1, hidden_dim=64, output_dim=1, num_layers=1)
model = model.to("cuda:0")
criterion = nn.L1Loss(reduction='mean')
# criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)


for epoch in range(args.epochs):
    model.train()
    losses = 0
    for dI, dR, dD, adj, result in myloader:
        optimizer.zero_grad()
        prediction = model(dI)
        batch_size, pred_window, num_node = result.shape
        result = result.reshape(batch_size*num_node, pred_window)
        loss = criterion(prediction, result)
        losses += loss.item()
        loss.backward()
        print(prediction)
        print(result)
        optimizer.step()
    print("Epoch: {:03d} | Train Loss: {:.3f} | lr = {:.10f}".format(epoch+1, losses/len(myloader), args.learn_rate))