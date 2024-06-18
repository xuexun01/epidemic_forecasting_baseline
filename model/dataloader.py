import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from utils import load_dict_npy_data
from torch.autograd import Variable


class JapanPrefecture(Dataset):
    def __init__(self, hist_seq_len, pred_seq_len, data_path, device, train_ratio, valid_ratio, mode):
        self.mode            = mode
        self.device          = device
        self.data_path       = data_path
        self.hist_seq_len    = hist_seq_len
        self.pred_seq_len    = pred_seq_len
        self.case_dict       = self._get_data(train_ratio, valid_ratio)
        self.hist, self.pred = self._generate_dataset()
        self.origin_adj, self.adj = self._get_adj()

    def _get_adj(self):
        origin_adj = torch.Tensor(np.loadtxt(open(f"{self.data_path}japan-adj.txt"), delimiter=',')).to(self.device)
        adj = Variable(origin_adj)
        return origin_adj, adj
    
    def _generate_dataset(self):
        hist = []
        pred = []
        for i in range(0, len(self.case_dict[self.mode])-self.hist_seq_len-self.pred_seq_len+1):
            hist.append(self.case_dict[self.mode][i:i+self.hist_seq_len])
            pred.append(self.case_dict[self.mode][i+self.hist_seq_len:i+self.hist_seq_len+self.pred_seq_len])
        return hist, pred
    
    def _get_data(self, train_ratio, valid_ratio):
        path = self.data_path + 'jp20200401_20210921.npy'
        data = np.load(path, allow_pickle='TRUE').item()
        cases = torch.tensor(data['node'][...,[0]], dtype=torch.float32).to(self.device)
        cases = np.squeeze(cases)
        case_dict = {}
        length = len(cases)
        case_dict['train'] = cases[:round(length*train_ratio)]
        case_dict['valid'] = cases[round(length*train_ratio):round(length*(train_ratio+valid_ratio))]
        case_dict['test']  = cases[round(length*(train_ratio+valid_ratio)):]
        return case_dict
    
    def __len__(self):
        return len(self.pred)
    
    def __getitem__(self, index):
        history_data, forecast_data = self.hist[index], self.pred[index]
        return history_data, forecast_data


class USState(Dataset):
    def __init__(self, hist_window, pred_window, data_path, dates, device):
        self.device = device
        self.data_path = data_path
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.raw_data = self.preprocess(dates)
        self.hist, self.pred = self.split_data()
        self.origin_adj, self.adj = self.get_adj()
    
    def get_adj(self):
        origin_adj = torch.Tensor(np.loadtxt(open(f"{self.data_path}state_adj.txt"), delimiter=',')).to(self.device)
        adj = Variable(origin_adj)
        return origin_adj, adj
    
    def preprocess(self, dates):
        state2fip = load_dict_npy_data(f"{self.data_path}state2fip.npy")
        state_order = list(state2fip.keys())

        confirmed = []
        dict = {}
        for date in dates:
            confirmed_df = pd.read_csv(f"{self.data_path}covid_19_daily_reports_us/{date}.csv").fillna(0)
            for row in confirmed_df.itertuples():
                if row.Province_State in state_order:
                    dict[row.Province_State] = row.Confirmed
            confirmed.append(list(dict.values()))
        confirmed = np.array(confirmed)
        new_cases = np.diff(confirmed, axis=0)
        return new_cases

    def split_data(self):
        hist = []
        pred = []
        for i in range(0, len(self.raw_data)-self.hist_window-self.pred_window+1):
            hist.append(self.raw_data[i:i+self.hist_window])
            pred.append(self.raw_data[i+self.hist_window:i+self.hist_window+self.pred_window])
        return hist, pred

    def __len__(self):
        return len(self.pred)
    
    def __getitem__(self, index):
        history_data, forecast_data = self.hist[index], self.pred[index]
        history_data = torch.tensor(history_data, dtype=torch.float32).to(self.device)
        forecast_data = torch.tensor(forecast_data, dtype=torch.float32).to(self.device)
        return history_data, forecast_data


if __name__ == "__main__":
    dates = pd.date_range(start="2020-04-12", end="2021-04-12")
    dates = list(dates.strftime("%Y_%m_%d"))
    dataset = USState(14, 14, "./data/", dates, device="cuda:0")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    i = 0
    for confirmed, y_true in loader:
        i += 1
        print(i)

    mydataset = JapanPrefecture(14, 14, "./data/", device="cuda:0", train_ratio=0.7, valid_ratio=0.1, mode='test')
    loader = DataLoader(mydataset, batch_size=32, shuffle=False)
    i = 0
    for preds, labels in loader:
        print(preds.shape)
        print(labels.shape)
        i += 1
        print(i)