import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from utils import set_environment


def get_data(dates, data_path, state):
    cases = []
    for date in dates:
        df = pd.read_csv(f"{data_path}covid_19_daily_reports_us/{date}.csv")
        cases.append(
            df.loc[df["Province_State"] == state, "Confirmed"].values[0]
        )
    return np.array(cases)


def preprocess_data(raw_data, hist_window, pred_window):
    data = []
    result = []
    for i in range(0, len(raw_data) - hist_window - pred_window + 1):
        data.append(raw_data[i : i + hist_window])
        result.append(raw_data[i + hist_window : i + hist_window + pred_window])
    return data, result


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        lstm_output, _ = self.lstm(x)
        output = self.fc(lstm_output[:, -1, :])
        output = F.relu(output)
        return output


class LSTM_attn(nn.Module):
    def __init__(self, hist_window, pred_window, hidden_dim, num_layers):
        super(LSTM_attn, self).__init__()
        self.lstm = nn.LSTM(hist_window, hidden_dim, num_layers)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=1)
        self.fc = nn.Linear(hidden_dim, pred_window)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(0)
        lstm_output, _ = self.lstm(x)
        attention_weights = self.attn(lstm_output)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        output = self.fc(context_vector)
        return output


if __name__ == "__main__":
    epochs = 100
    learn_rate = 0.001
    hist_window = 4
    pred_window = 1

    set_environment(seed=115327)
    dates = pd.date_range(start="2020-04-12", end="2021-10-12")
    dates = list(dates.strftime("%Y_%m_%d"))
    raw_data = get_data(dates, "./data/", state="Alabama")
    sns.lineplot(x=[x for x in range(1,len(raw_data)+1)], y=raw_data)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # raw_data = scaler.fit_transform(raw_data.reshape(-1, 1))

    data, result = preprocess_data(raw_data, hist_window, pred_window)
    data = torch.tensor(data, dtype=torch.float32)
    result = torch.tensor(result, dtype=torch.float32)

    model = LSTM(input_dim=1, hidden_dim=64, output_dim=pred_window, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    for epoch in range(epochs):
        model.train()
        prediction = model(data)
        loss = criterion(prediction, result.unsqueeze(-1))
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        print("Epoch: {:03d} | Train Loss: {:.10f} | lr = {:.6f}".format(epoch + 1, loss.item(), learn_rate))
    
    # prediction = scaler.inverse_transform(prediction.detach().numpy())
    # prediction = prediction.flatten().tolist()
    prediction = prediction.detach().numpy().flatten().tolist()
    sns.lineplot(x=[x+hist_window+1 for x in range(len(prediction))], y=prediction)
    plt.show()