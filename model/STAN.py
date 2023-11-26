import dgl
import pandas as pd
import numpy as np
from haversine import haversine
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))
        
class STAN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_heads, pred_window, device):
        super(STAN, self).__init__()
        self.g = g
        
        self.layer1 = MultiHeadGATLayer(self.g, in_dim, hidden_dim1, num_heads)
        self.layer2 = MultiHeadGATLayer(self.g, hidden_dim1 * num_heads, hidden_dim2, 1)

        self.pred_window = pred_window
        self.gru = nn.GRUCell(hidden_dim2, gru_dim)
    
        self.nn_res_I = nn.Linear(gru_dim+2, pred_window)
        self.nn_res_R = nn.Linear(gru_dim+2, pred_window)

        self.nn_res_sir = nn.Linear(gru_dim+2, 2)
        
        self.hidden_dim2 = hidden_dim2
        self.gru_dim = gru_dim
        self.device = device

    def forward(self, dynamic, cI, cR, N, I, R, h=None):
        num_loc, timestep, n_feat = dynamic.size()
        # 总人数，一维张量
        N = N.squeeze()

        if h is None:
            h = torch.zeros(1, self.gru_dim).to(self.device)
            gain = nn.init.calculate_gain("relu")
            nn.init.xavier_normal_(h, gain=gain)

        new_I = []
        new_R = []
        phy_I = []
        phy_R = []
        self.alpha_list = []
        self.beta_list = []
        self.alpha_scaled = []
        self.beta_scaled = []

        for each_step in range(timestep):
            cur_h = self.layer1(dynamic[:, each_step, :])
            cur_h = F.elu(cur_h)
            cur_h = self.layer2(cur_h)
            cur_h = F.elu(cur_h)

            # 最大池化，选择每个特征维度上的最大值，并将结果重塑为形状 (1, self.hidden_dim2) 的张量
            cur_h = torch.max(cur_h, 0)[0].reshape(1, self.hidden_dim2)

            # 更新隐藏状态 h
            h = self.gru(cur_h, h)
            hc = torch.cat(
                (h, cI[each_step].reshape(1, 1), cR[each_step].reshape(1, 1)), dim=1
            )

            pred_I = self.nn_res_I(hc)
            pred_R = self.nn_res_R(hc)
            new_I.append(pred_I)
            new_R.append(pred_R)

            pred_res = self.nn_res_sir(hc)
            alpha = pred_res[:, 0]
            beta = pred_res[:, 1]

            self.alpha_list.append(alpha)
            self.beta_list.append(beta)
            alpha = torch.sigmoid(alpha)
            beta = torch.sigmoid(beta)
            self.alpha_scaled.append(alpha)
            self.beta_scaled.append(beta)

            cur_phy_I = []
            cur_phy_R = []
            # 循环预测未来的感染者和康复者数量
            for i in range(self.pred_window):
                last_I = I[each_step] if i == 0 else last_I + dI.detach()
                last_R = R[each_step] if i == 0 else last_R + dR.detach()

                last_S = N - last_I - last_R

                dI = alpha * last_I * (last_S / N) - beta * last_I
                dR = beta * last_I
                cur_phy_I.append(dI)
                cur_phy_R.append(dR)
            cur_phy_I = torch.stack(cur_phy_I).to(self.device).permute(1, 0)
            cur_phy_R = torch.stack(cur_phy_R).to(self.device).permute(1, 0)

            phy_I.append(cur_phy_I)
            phy_R.append(cur_phy_R)

        # 维度重排
        new_I = torch.stack(new_I).to(self.device).permute(1, 0, 2)
        new_R = torch.stack(new_R).to(self.device).permute(1, 0, 2)
        phy_I = torch.stack(phy_I).to(self.device).permute(1, 0, 2)
        phy_R = torch.stack(phy_R).to(self.device).permute(1, 0, 2)

        self.alpha_list = torch.stack(self.alpha_list).squeeze()
        self.beta_list = torch.stack(self.beta_list).squeeze()
        self.alpha_scaled = torch.stack(self.alpha_scaled).squeeze()
        self.beta_scaled = torch.stack(self.beta_scaled).squeeze()
        return new_I, new_R, phy_I, phy_R, h


    

# 引力定律相关性
def gravity_law_commute_dist(lat1, lng1, pop1, lat2, lng2, pop2, r=1):
    d = haversine((lat1, lng1), (lat2, lng2), "km")
    c = 1
    w = 0
    alpha = 0.1
    beta = 0.1
    r = 1e4
    w = (np.exp(-d / r)) / (abs((pop1**alpha) - (pop2**beta)) + 1e-5)
    return w


def prepare_data(data, sum_I, sum_R, history_window=5, pred_window=15, slide_step=5):
    # Data shape n_loc, timestep, n_feat
    # Reshape to n_loc, t, history_window*n_feat
    n_loc = data.shape[0]
    timestep = data.shape[1]
    n_feat = data.shape[2]

    x = []
    y_I = []
    y_R = []
    last_I = []
    last_R = []
    concat_I = []
    concat_R = []
    for i in range(0, timestep, slide_step):
        if (i + history_window + pred_window - 1 >= timestep or i + history_window >= timestep):
            break
        x.append(data[:, i : i + history_window, :].reshape((n_loc, history_window * n_feat)))

        concat_I.append(data[:, i + history_window - 1, 0])
        concat_R.append(data[:, i + history_window - 1, 1])
        last_I.append(sum_I[:, i + history_window - 1])
        last_R.append(sum_R[:, i + history_window - 1])

        y_I.append(data[:, i + history_window : i + history_window + pred_window, 0])
        y_R.append(data[:, i + history_window : i + history_window + pred_window, 1])

    x = np.array(x, dtype=np.float32).transpose((1, 0, 2))
    last_I = np.array(last_I, dtype=np.float32).transpose((1, 0))
    last_R = np.array(last_R, dtype=np.float32).transpose((1, 0))
    concat_I = np.array(concat_I, dtype=np.float32).transpose((1, 0))
    concat_R = np.array(concat_R, dtype=np.float32).transpose((1, 0))
    y_I = np.array(y_I, dtype=np.float32).transpose((1, 0, 2))
    y_R = np.array(y_R, dtype=np.float32).transpose((1, 0, 2))
    return x, last_I, last_R, concat_I, concat_R, y_I, y_R


data_path = "./data/"
pop_data = pd.read_csv(f"{data_path}state_info.csv")


state_list = list(pop_data["State"])
loc_dist_map = {}
for origin in state_list:
    loc_dist_map[origin] = {}
    for destination in state_list:
        lat_o = pop_data[pop_data["State"] == origin]["Latitude"].unique()[0]
        lng_o = pop_data[pop_data["State"] == origin]["Longitude"].unique()[0]
        pop_o = int(pop_data[pop_data["State"] == origin]["Resident Population"].unique()[0].replace(',', ''))
        lat_d = pop_data[pop_data["State"] == destination]["Latitude"].unique()[0]
        lng_d = pop_data[pop_data["State"] == destination]["Longitude"].unique()[0]
        pop_d = int(pop_data[pop_data["State"] == destination]["Resident Population"].unique()[0].replace(',', ''))
        loc_dist_map[origin][destination] = gravity_law_commute_dist(lat_o, lng_o, pop_o, lat_d, lng_d, pop_d, r=0.5)


static_feat = []
for state in state_list:
        item = pop_data[pop_data["State"] == state]
        population = int(item["Resident Population"].values[0].replace(',', ''))
        density = float(item["Population Density"].values[0].replace(',', ''))
        lng = item["Longitude"].values[0]
        lat = item["Latitude"].values[0]
        static_feat.append(np.array([population, density, lng, lat]))
static_feat = np.vstack(static_feat)


# 对地点距离进行排序
dist_threshold = 18
for each_loc in loc_dist_map:
    loc_dist_map[each_loc] = {k: v for k, v in sorted(loc_dist_map[each_loc].items(), key=lambda item: item[1], reverse=True)}

# 构建邻接矩阵
# 如果地点之间的距离大于阈值 dist_threshold，则将其添加到邻接地点列表中。根据距离的排序，最多添加4个远距离地点和2个近距离地点。
adj_map = {}
for each_loc in loc_dist_map:
    adj_map[each_loc] = []
    for i, each_loc2 in enumerate(loc_dist_map[each_loc]):
        if loc_dist_map[each_loc][each_loc2] > dist_threshold:
            if i <= 3:
                adj_map[each_loc].append(each_loc2)
            else:
                break
        else:
            if i <= 1:
                adj_map[each_loc].append(each_loc2)
            else:
                break

rows, cols = [], []
for each_loc in adj_map:
    for each_loc2 in adj_map[each_loc]:
        rows.append(state_list.index(each_loc))
        cols.append(state_list.index(each_loc2))

# 生成图
graph = dgl.graph((rows, cols))


# 获取SIR数据
dates = pd.date_range(start='2020-05-01', end='2020-12-01')
dates = list(dates.strftime("%Y_%m_%d"))


data = []
for date in dates:
    df = pd.read_csv(f"{data_path}covid_19_daily_reports_us/{date}.csv")
    df.loc[:, "date_today"] = datetime.strptime(date, "%Y_%m_%d")
    df = df.rename(
        columns={
            "Province_State": "state",
            "Confirmed": "confirmed",
            "Deaths": "deaths",
            "Recovered": "recovered",
            "Active": "active",
        }
    )
    data.append(df)

data = [x for x in data if x is not None]
data = pd.concat(data, axis=0).ffill(axis=0).fillna(0)
data.loc[:, "date_today"] = pd.to_datetime(data["date_today"])
df = []
for state in data["state"].unique():
    temp = data[data["state"] == state].sort_values("date_today")
    temp.loc[:, "new_cases"] = temp["confirmed"].copy()
    # transform to daily cases
    t = temp["new_cases"].copy().to_numpy()
    t[1:] = t[1:] - t[:-1]
    temp = temp.iloc[1:]
    t[t < 0] = 0
    temp.loc[:, "new_cases"] = t[1:]
    df.append(temp)
df = pd.concat(df, axis=0)


active_cases = []
confirmed_cases = []
new_cases = []
death_cases = []

for i, each_loc in enumerate(state_list):
    active_cases.append(df[df["state"] == each_loc]["active"])
    confirmed_cases.append(df[df["state"] == each_loc]["confirmed"])
    new_cases.append(df[df["state"] == each_loc]["new_cases"])
    death_cases.append(df[df["state"] == each_loc]["deaths"])


active_cases = np.array(active_cases)
confirmed_cases = np.array(confirmed_cases)
death_cases = np.array(death_cases)
new_cases = np.array(new_cases)
recovered_cases = confirmed_cases - active_cases - death_cases
susceptible_cases = np.expand_dims(static_feat[:, 0], -1) - active_cases - recovered_cases


dI = np.concatenate((np.zeros((active_cases.shape[0], 1), dtype=np.float32), np.diff(active_cases)), axis=-1)
dR = np.concatenate((np.zeros((recovered_cases.shape[0], 1), dtype=np.float32), np.diff(recovered_cases)), axis=-1)
dS = np.concatenate((np.zeros((susceptible_cases.shape[0], 1), dtype=np.float32), np.diff(susceptible_cases)), axis=-1)



normalizer = {"S": {}, "I": {}, "R": {}, "dS": {}, "dI": {}, "dR": {}}
for i, each_loc in enumerate(state_list):
    normalizer["S"][each_loc] = (np.mean(susceptible_cases[i]), np.std(susceptible_cases[i]))
    normalizer["I"][each_loc] = (np.mean(active_cases[i]), np.std(active_cases[i]))
    normalizer["R"][each_loc] = (np.mean(recovered_cases[i]), np.std(recovered_cases[i]))
    normalizer["dI"][each_loc] = (np.mean(dI[i]), np.std(dI[i]))
    normalizer["dR"][each_loc] = (np.mean(dR[i]), np.std(dR[i]))
    normalizer["dS"][each_loc] = (np.mean(dS[i]), np.std(dS[i]))


dynamic_feat = np.concatenate((np.expand_dims(dI, axis=-1), np.expand_dims(dR, axis=-1), np.expand_dims(dS, axis=-1)), axis=-1)
for i, each_loc in enumerate(state_list):
    dynamic_feat[i, :, 0] = (dynamic_feat[i, :, 0] - normalizer["dI"][each_loc][0]) / normalizer["dI"][each_loc][1]
    dynamic_feat[i, :, 1] = (dynamic_feat[i, :, 1] - normalizer["dR"][each_loc][0]) / normalizer["dR"][each_loc][1]
    dynamic_feat[i, :, 2] = (dynamic_feat[i, :, 2] - normalizer["dS"][each_loc][0]) / normalizer["dS"][each_loc][1]


dI_mean = []
dI_std = []
dR_mean = []
dR_std = []
valid_window = 25
test_window = 25
history_window = 6
pred_window = 15
slide_step = 5

for i, each_loc in enumerate(state_list):
    dI_mean.append(normalizer["dI"][each_loc][0])
    dR_mean.append(normalizer["dR"][each_loc][0])
    dI_std.append(normalizer["dI"][each_loc][1])
    dR_std.append(normalizer["dR"][each_loc][1])

dI_mean = np.array(dI_mean)
dI_std = np.array(dI_std)
dR_mean = np.array(dR_mean)
dR_std = np.array(dR_std)

# Split train-test
train_feat = dynamic_feat[:, : -valid_window - test_window, :]
val_feat = dynamic_feat[:, -valid_window - test_window : -test_window, :]
test_feat = dynamic_feat[:, -test_window:, :]

train_x, train_I, train_R, train_cI, train_cR, train_yI, train_yR = prepare_data(train_feat, active_cases[:, : -valid_window - test_window], 
                                                                                 recovered_cases[:, : -valid_window - test_window], history_window, pred_window, slide_step)
val_x, val_I, val_R, val_cI, val_cR, val_yI, val_yR = prepare_data(val_feat, active_cases[:, -valid_window - test_window : -test_window],
                                                                   recovered_cases[:, -valid_window - test_window : -test_window], history_window, pred_window, slide_step)
test_x, test_I, test_R, test_cI, test_cR, test_yI, test_yR = prepare_data(test_feat, active_cases[:, -test_window:], recovered_cases[:, -test_window:], history_window, pred_window, slide_step)

in_dim = 3 * history_window
hidden_dim1 = 32
hidden_dim2 = 32
gru_dim = 32
num_heads = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

graph = graph.to(device)
model = STAN(graph, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_heads, pred_window, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()

train_x = torch.tensor(train_x).to(device)
train_I = torch.tensor(train_I).to(device)
train_R = torch.tensor(train_R).to(device)
train_cI = torch.tensor(train_cI).to(device)
train_cR = torch.tensor(train_cR).to(device)
train_yI = torch.tensor(train_yI).to(device)
train_yR = torch.tensor(train_yR).to(device)

val_x = torch.tensor(val_x).to(device)
val_I = torch.tensor(val_I).to(device)
val_R = torch.tensor(val_R).to(device)
val_cI = torch.tensor(val_cI).to(device)
val_cR = torch.tensor(val_cR).to(device)
val_yI = torch.tensor(val_yI).to(device)
val_yR = torch.tensor(val_yR).to(device)

test_x = torch.tensor(test_x).to(device)
test_I = torch.tensor(test_I).to(device)
test_R = torch.tensor(test_R).to(device)
test_cI = torch.tensor(test_cI).to(device)
test_cR = torch.tensor(test_cR).to(device)
test_yI = torch.tensor(test_yI).to(device)
test_yR = torch.tensor(test_yR).to(device)

dI_mean = torch.tensor(dI_mean, dtype=torch.float32).to(device).reshape((dI_mean.shape[0], 1, 1))
dI_std = torch.tensor(dI_std, dtype=torch.float32).to(device).reshape((dI_mean.shape[0], 1, 1))
dR_mean = torch.tensor(dR_mean, dtype=torch.float32).to(device).reshape((dI_mean.shape[0], 1, 1))
dR_std = torch.tensor(dR_std, dtype=torch.float32).to(device).reshape((dI_mean.shape[0], 1, 1))
N = torch.tensor(static_feat[:, 0], dtype=torch.float32).to(device).unsqueeze(-1)

# Train STAN
all_loss = []
file_name = "./pt/stan"
min_loss = 1e10

loc_name = 'Kentucky'
cur_loc = state_list.index(loc_name)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    active_pred, recovered_pred, phy_active, phy_recover, _ = model(
        train_x,
        train_cI[cur_loc],
        train_cR[cur_loc],
        N[cur_loc],
        train_I[cur_loc],
        train_R[cur_loc],
    )
    phy_active = (phy_active - dI_mean[cur_loc]) / dI_std[cur_loc]
    phy_recover = (phy_recover - dR_mean[cur_loc]) / dR_std[cur_loc]
    loss = (criterion(active_pred.squeeze(), train_yI[cur_loc])
        + criterion(recovered_pred.squeeze(), train_yR[cur_loc])
        + 0.1 * criterion(phy_active.squeeze(), train_yI[cur_loc])
        + 0.1 * criterion(phy_recover.squeeze(), train_yR[cur_loc])
    )

    loss.backward()
    optimizer.step()
    all_loss.append(loss.item())

    model.eval()
    _, _, _, _, prev_h = model(train_x, train_cI[cur_loc], train_cR[cur_loc], N[cur_loc], train_I[cur_loc], train_R[cur_loc])
    val_active_pred, val_recovered_pred, val_phy_active, val_phy_recover, _ = model(val_x, val_cI[cur_loc], val_cR[cur_loc], N[cur_loc], val_I[cur_loc], val_R[cur_loc], prev_h)

    val_phy_active = (val_phy_active - dI_mean[cur_loc]) / dI_std[cur_loc]
    val_loss = criterion(val_active_pred.squeeze(), val_yI[cur_loc]) + 0.1 * criterion(val_phy_active.squeeze(), val_yI[cur_loc])
    if val_loss < min_loss:
        state = {"state": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(state, file_name)
        min_loss = val_loss
        print("-----Save best model-----")

    print("Epoch %d, Loss %.2f, Val loss %.2f" % (epoch, all_loss[-1], val_loss.item()))


checkpoint = torch.load(file_name)
model.load_state_dict(checkpoint["state"])
optimizer.load_state_dict(checkpoint["optimizer"])
model.eval()
prev_x = torch.cat((train_x, val_x), dim=1)
prev_I = torch.cat((train_I, val_I), dim=1)
prev_R = torch.cat((train_R, val_R), dim=1)
prev_cI = torch.cat((train_cI, val_cI), dim=1)
prev_cR = torch.cat((train_cR, val_cR), dim=1)


prev_active_pred, _, prev_phyactive_pred, _, h = model(prev_x, prev_cI[cur_loc], prev_cR[cur_loc], N[cur_loc], prev_I[cur_loc], prev_R[cur_loc])
test_pred_active, test_pred_recovered, test_pred_phy_active, test_pred_phy_recover, _ = model(test_x, test_cI[cur_loc], test_cR[cur_loc], N[cur_loc], test_I[cur_loc], test_R[cur_loc], h)
print("Estimated beta in SIR model is %.2f" % model.alpha_scaled)
print("Estimated gamma in SIR model is %.2f" % model.beta_scaled)

pred_I = []

for i in range(test_pred_active.size(1)):
    cur_pred = test_pred_active[0, i, :].detach().cpu().numpy() * dI_std[cur_loc].reshape(1, 1).detach().cpu().numpy() + dI_mean[cur_loc].reshape(1, 1).detach().cpu().numpy()
    # cur_pred = test_pred_phy_active[0, i, :].detach().cpu().numpy()
    cur_pred = (cur_pred + test_pred_phy_active[0, i, :].detach().cpu().numpy()) / 2
    cur_pred = np.cumsum(cur_pred)
    cur_pred = cur_pred + test_I[cur_loc, i].detach().cpu().item()
    pred_I.append(cur_pred)
pred_I = np.array(pred_I)
pred_I = pred_I

def get_real_y(data, history_window=5, pred_window=15, slide_step=5):
    # Data shape n_loc, timestep, n_feat
    # Reshape to n_loc, t, history_window*n_feat
    n_loc = data.shape[0]
    timestep = data.shape[1]

    y = []
    for i in range(0, timestep, slide_step):
        if (
            i + history_window + pred_window - 1 >= timestep
            or i + history_window >= timestep
        ):
            break
        y.append(data[:, i + history_window : i + history_window + pred_window])
    y = np.array(y, dtype=np.float32).transpose((1, 0, 2))
    return y


I_true = get_real_y(active_cases[:], history_window, pred_window, slide_step)
plt.plot(I_true[cur_loc, -1, :], c="r", label="Ground truth")
plt.plot(pred_I[-1, :], c="b", label="Prediction")
plt.legend()
plt.show()