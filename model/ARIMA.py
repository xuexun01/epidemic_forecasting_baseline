import argparse
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from utils import *
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("--hist_window", type=int, default=14)
parser.add_argument("--pred_window", type=int, default=3)
parser.add_argument("--dataset", type=str, default="japan")
args = parser.parse_args()


def get_data(dates, data_path, state):
    confirmed = []
    for date in dates:
        df = pd.read_csv(f"{data_path}covid_19_daily_reports_us/{date}.csv")
        confirmed.append(df.loc[df["Province_State"] == state, "Confirmed"].values[0])
    confirmed = np.array(confirmed)
    new_cases = np.diff(confirmed, axis=0)
    return new_cases


def create_time_series(data, hist_window, pred_window):
    X, y = [], []
    for i in range(len(data) - hist_window - pred_window + 1):
        X.append(data[i:i+hist_window])
        y.append(data[i+hist_window: i+hist_window+pred_window])
    return np.array(X), np.array(y)



data_path = "./data/"
if args.dataset == "us":
    pop_data = pd.read_csv(f"{data_path}state_info.csv")
    state_list = list(pop_data["State"])

    dates = pd.date_range(start="2020-04-12", end="2021-04-15")
    dates = list(dates.strftime("%Y_%m_%d"))
    data = get_data(dates, data_path, state="California")
else:
    raw_data = np.load(f"{data_path}jp20200401_20210921.npy", allow_pickle='TRUE').item()
    data = raw_data['node'][:, 0,[0]]

# scaler = MinMaxScaler(feature_range=(0, 1))
# data_scaled = scaler.fit_transform(data.reshape(-1, 1))
# train_size = int(len(data_scaled) * 0.8)
# train_data = data_scaled[:train_size, :]
# test_data = data_scaled[train_size:, :]
    
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]
X_test, y_test = create_time_series(test_data, args.hist_window, args.pred_window)


total_rmse = 0
total_mae = 0
total_mape = 0
for inputs, labels in zip(X_test, y_test):
    train_data = np.concatenate((train_data[:-args.hist_window], inputs), axis=0)
    model = ARIMA(train_data, order=(args.hist_window, 1, 1))
    model_fit = model.fit()
    preds = model_fit.forecast(steps=args.pred_window)
    # preds = scaler.inverse_transform(preds.reshape((1, -1)))
    # labels = scaler.inverse_transform(labels.reshape((1, -1)))
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = np.mean(np.abs(labels - preds))
    mape = np.mean(np.abs(labels - preds) / labels)
    print("[Test] RMSE: {} \t MAE: {} \t MAPE: {}".format(rmse, mae, mape))
    total_rmse += rmse
    total_mae += mae
    total_mape += mape

total_rmse = total_rmse/X_test.shape[0]
total_mae = total_mae/X_test.shape[0]
total_mape = total_mape/X_test.shape[0]
print(f"Total RMSE: {total_rmse} \t MAE: {total_mae} \t MAPE: {total_mape}")