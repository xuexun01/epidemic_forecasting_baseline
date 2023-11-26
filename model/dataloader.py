import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from utils import load_dict_npy_data


class CCDataset(Dataset):
    def __init__(self, hist_window, pred_window, data_path, dates, device):
        self.device = device
        self.data_path = data_path
        self.hist_window = hist_window
        self.pred_window = pred_window
        self.dates = dates
        self.data_indexs = self._get_data()
    
    def _get_data(self):
        data_indexs = []
        for i in range(0, len(self.dates)-self.hist_window-self.pred_window+1):
            data_indexs.append(self.dates[i:i+self.hist_window+self.pred_window])
        return data_indexs
    
    def _load_preprocess_data(self, data_index):
        state_info = pd.read_csv(f"{self.data_path}state_info.csv", index_col=0)
        state2fip = load_dict_npy_data(f"{self.data_path}state2fip.npy")
        fip2state = load_dict_npy_data(f"{self.data_path}fip2state.npy")
        state_order = list(state2fip.keys())
        fip_order = list(state2fip.values())
        
        # load raw data
        case_dfs = []
        mobility_dfs = []
        for date in data_index:
            case_dfs.append(pd.read_csv(f"{self.data_path}covid_19_daily_reports_us/{date}.csv").fillna(0))
            mobility_dfs.append(pd.read_csv(f"{self.data_path}state2state/daily_state2state_{date}.csv"))
        
        # preprocess data
        N, S, I, R, D, mob, y_true = [], [], [], [], [], [], []
        # history window
        for i in range(0, self.hist_window):
            # epidemic data
            s_dict = {}
            i_dict = {}
            r_dict = {}
            d_dict = {}
            n_dict = {}
            for row in case_dfs[i].itertuples():
                if row.Province_State in state_order:
                    pop = int(state_info.loc[row.Province_State, "Resident Population"].replace(',', ''))
                    n_dict[row.Province_State] = pop
                    s_dict[row.Province_State] = pop - row.Confirmed
                    i_dict[row.Province_State] = row.Confirmed - row.Recovered - row.Deaths
                    r_dict[row.Province_State] = row.Recovered
                    d_dict[row.Province_State] = row.Deaths
            N.append(list(n_dict.values()))
            S.append(list(s_dict.values()))
            I.append(list(i_dict.values()))
            R.append(list(r_dict.values()))
            D.append(list(d_dict.values()))

            # mobility data
            graph = nx.DiGraph()
            graph.add_nodes_from(fip_order)
            edges_with_weights = []
            for row in mobility_dfs[i].itertuples():
                geoid_o = str(row.geoid_o).zfill(2)
                geoid_d = str(row.geoid_d).zfill(2)
                if geoid_o in fip_order and geoid_d in fip_order:
                    pop = int(state_info.loc[fip2state[geoid_o], "Resident Population"].replace(',', ''))
                    edges_with_weights.append((geoid_o, geoid_d, row.pop_flows/pop))
            graph.add_weighted_edges_from(edges_with_weights)
            mob.append(nx.to_numpy_array(graph, weight='weight'))
        
        # prediction window
        for i in range(self.hist_window, self.hist_window+self.pred_window):
            daily_confirmed = {}
            for row in case_dfs[i].itertuples():
                if row.Province_State in state_order:
                    daily_confirmed[row.Province_State] = row.Confirmed
            y_true.append(list(daily_confirmed.values()))

        return N, S, I, R, D, mob, y_true

    def __len__(self):
        return len(self.data_indexs)

    def __getitem__(self, index):
        data_index = self.data_indexs[index]
        N, S, I, R, D, mob, y_true = self._load_preprocess_data(data_index)
        N = torch.tensor(N, dtype=torch.float32).to(self.device)
        S = torch.tensor(S, dtype=torch.float32).to(self.device)
        I = torch.tensor(I, dtype=torch.float32).to(self.device)
        R = torch.tensor(R, dtype=torch.float32).to(self.device)
        D = torch.tensor(D, dtype=torch.float32).to(self.device)
        mob = torch.from_numpy(np.stack(mob)).to(self.device, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.float32).to(self.device)
        return N, S, I, R, D, mob, y_true


if __name__ == "__main__":

    dates = pd.date_range(start="2020-04-12", end="2021-04-12")
    dates = list(dates.strftime("%Y_%m_%d"))
    mydataset = CCDataset(14, 14, "./data/", dates, device="cuda:0")
    myloader = DataLoader(mydataset, batch_size=32, shuffle=False)
    for S, I, R, D, mob, y_true in myloader:
        print(S.shape)
        print(I.shape)
        print(R.shape)
        print(D.shape)
        print(mob.shape)
        print(y_true.shape)
        break
        # if torch.any(dR == 0):
        #     print("dR中存在0")
        #     print(torch.sum(torch.eq(dR, 0)))