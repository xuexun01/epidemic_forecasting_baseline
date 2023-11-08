import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import networkx as nx
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
        pop_dict = load_dict_npy_data(f"{self.data_path}pop.npy")
        state2fip = load_dict_npy_data(f"{self.data_path}state2fip.npy")
        state_order = list(state2fip.keys())
        fip_order = list(state2fip.values())
        
        # load raw data
        case_dfs = []
        mobility_dfs = []
        for date in data_index:
            case_dfs.append(pd.read_csv(f"{self.data_path}covid_19_daily_reports_us/{date}.csv"))
            mobility_dfs.append(pd.read_csv(f"{self.data_path}state2state/daily_state2state_{date}.csv"))
        
        # preprocess data
        dI, dR, dD, adj, result = [], [], [], [], []
        # history window
        for i in range(0, self.hist_window):
            dI_dict = {}
            dR_dict = {}
            dD_dict = {}
            for row in case_dfs[i].itertuples():
                if row.Province_State in state_order:
                    dI_dict[row.Province_State] = row.Confirmed
                    dR_dict[row.Province_State] = row.Recovered
                    dD_dict[row.Province_State] = row.Deaths
            dI.append(list(dI_dict.values()))
            dR.append(list(dR_dict.values()))
            dD.append(list(dD_dict.values()))

            graph = nx.DiGraph()
            graph.add_nodes_from(fip_order)
            edges_with_weights = []
            for row in mobility_dfs[i].itertuples():
                geoid_o = str(row.geoid_o).zfill(2)
                geoid_d = str(row.geoid_d).zfill(2)
                if geoid_o in fip_order and geoid_d in fip_order:
                    edges_with_weights.append((geoid_o, geoid_d, row.pop_flows/pop_dict[geoid_o]))
            graph.add_weighted_edges_from(edges_with_weights)
            adj.append(nx.to_numpy_array(graph, weight='weight'))
        
        # prediction window
        for i in range(self.hist_window, self.hist_window+self.pred_window):
            daily_confirmed = {}
            for row in case_dfs[i].itertuples():
                if row.Province_State in state_order:
                    daily_confirmed[row.Province_State] = row.Confirmed
            result.append(list(daily_confirmed.values()))

        return dI, dR, dD, adj, result

    def __len__(self):
        return len(self.data_indexs)

    def __getitem__(self, index):
        data_index = self.data_indexs[index]
        dI, dR, dD, adj, result = self._load_preprocess_data(data_index)
        dI = torch.tensor(dI, dtype=torch.float32).to(self.device)
        dR = torch.tensor(dR, dtype=torch.float32).to(self.device)
        dD = torch.tensor(dD, dtype=torch.float32).to(self.device)
        adj = torch.from_numpy(np.stack(adj)).to(self.device)
        result = torch.tensor(result, dtype=torch.float32).to(self.device)
        return dI, dR, dD, adj, result




if __name__ == "__main__":

    dates = pd.date_range(start="2021-01-01", end="2021-04-15")
    dates = list(dates.strftime("%Y_%m_%d"))
    mydataset = CCDataset(12, 2, "./data/", dates, device="cpu")
    myloader = DataLoader(mydataset, batch_size=6, shuffle=False)
    for dI, dR, dD, adj, result in myloader:
        print(dI.shape)
        print(dR.shape)
        print(dD.shape)
        print(adj.shape)
        print(result.shape)
        print(dI[0,0,:])
        break