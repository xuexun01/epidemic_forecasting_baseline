import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Attention, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model", type=str, default="lstm")
parser.add_argument("--hidden_size", type=int, default=32)
parser.add_argument("--hist_window", type=int, default=14)
parser.add_argument("--pred_window", type=int, default=1)
parser.add_argument("--dataset", type=str, default="us")
args = parser.parse_args()


np.random.seed(3407)
tf.random.set_seed(3407)

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
        X.append(data[i:i+hist_window, 0])
        y.append(data[i+hist_window: i+hist_window+pred_window, 0])
    return np.array(X), np.array(y)

if args.dataset == "us":
    dates = pd.date_range(start="2020-04-12", end="2021-04-15")
    dates = list(dates.strftime("%Y_%m_%d"))
    raw_data = get_data(dates, "./data/", state="Alabama")
else:
    data_path = "./data/"
    raw_data = np.load(f"{data_path}jp20200401_20210921.npy", allow_pickle='TRUE').item()
    raw_data = raw_data['node'][:, 0,[0]]

# scaler = MinMaxScaler(feature_range=(0, 1))
# data_scaled = scaler.fit_transform(raw_data.reshape(-1, 1))
# train_size = int(len(data_scaled) * 0.8)
# train_data = data_scaled[:train_size, :]
# test_data = data_scaled[train_size:, :]

raw_data = raw_data.reshape(-1, 1)
train_size = int(len(raw_data) * 0.8)
train_data = raw_data[:train_size, :]
test_data = raw_data[train_size:, :]

X_train, y_train = create_time_series(train_data, args.hist_window, args.pred_window)
X_test, y_test = create_time_series(test_data, args.hist_window, args.pred_window)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

if args.model == "lstm":
    model = Sequential()
    model.add(LSTM(args.hidden_size, activation='relu', input_shape=(args.hist_window, 1)))
    model.add(Dense(args.pred_window))
    model.compile(optimizer='adam', loss='mean_squared_error')
else:
    input_layer = Input(shape=(args.hist_window, 1))
    lstm1 = LSTM(args.hidden_size, activation='relu', return_sequences=True)(input_layer)
    attention = Attention()([lstm1, lstm1])
    lstm2 = LSTM(args.hidden_size, activation='relu', return_sequences=True)(attention)
    attention2 = Attention()([lstm2, lstm2])
    lstm3 = LSTM(args.hidden_size, activation='relu')(attention2)
    output_layer = Dense(args.pred_window, activation='sigmoid', name='output')(lstm3)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')

# train
model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

# test the model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# train_predict = scaler.inverse_transform(train_predict)
# y_train = scaler.inverse_transform(y_train)
# test_predict = scaler.inverse_transform(test_predict)
# y_test = scaler.inverse_transform(y_test)

# calculate RMSE
train_rmse = np.sqrt(mean_squared_error(train_predict, y_train))

test_rmse = np.sqrt(mean_squared_error(test_predict, y_test))
test_mae = np.mean(np.abs(y_test - test_predict))
test_mape = np.mean(np.abs((y_test - test_predict)) / y_test)


print('Train RMSE:', train_rmse)
print("[Test loss] RMSE: {} \t MAE: {} \t MAPE: {}".format(test_rmse, test_mae, test_mape))