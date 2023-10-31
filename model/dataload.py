import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def load_data(datapath):
    pop_dict = np.load(f"{datapath}pop.npy", allow_pickle=True).item()
    commute_array = np.load(f"{datapath}commute.npy")
    confirmed_dict = np.load(f"{datapath}confirmed.npy", allow_pickle=True).item()
    deaths_dict = np.load(f"{datapath}deaths.npy", allow_pickle=True).item()
    return pop_dict, commute_array, confirmed_dict, deaths_dict


def get_flows_mobility(datapath, dates, batch_size):
    for i in range(0, len(dates) + 1 - batch_size, batch_size):
        dates_sub = dates[i : i + batch_size]
        flows_mobility = []
        for date in dates_sub:
            print(f"{datapath}daily_county2county_{date}.csv")
            filename = f"{datapath}daily_county2county_{date}.csv"
            df = pd.read_csv(filename)
            df = df[["geoid_o", "geoid_d", "pop_flows"]]
            flows_mobility.append(df)
        yield flows_mobility


class FlowsDataset(Dataset):
    def __init__(self, hist_window, pred_window, datapath):
        self.hist_window = hist_window
        self.pred_window = pred_window
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


if __name__ == "__main__":
    datapath = "./data/"

    dates = pd.date_range(start="2020-03-15", end="2021-04-15")
    dates = list(dates.strftime("%Y_%m_%d"))
    flows_datapath = datapath + "county2county/"
    pop_dict, commute_array, confirmed_dict, deaths_dict = load_data(datapath)
    for dfs in get_flows_mobility(flows_datapath, dates, 1):
        print(len(dfs))