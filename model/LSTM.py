import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Attention, Input
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learn_rate", type=float, default=0.001)
parser.add_argument("--model", type=str, default="lstm")
parser.add_argument("--hidden_size", type=int, default=50)
parser.add_argument("--hist_window", type=int, default=10)
parser.add_argument("--pred_window", type=int, default=1)
args = parser.parse_args()


def get_data(dates, data_path, state):
    cases = []
    for date in dates:
        df = pd.read_csv(f"{data_path}covid_19_daily_reports_us/{date}.csv")
        cases.append(df.loc[df["Province_State"] == state, "Confirmed"].values[0])
    return np.array(cases)


def create_time_series(data, hist_window, pred_window):
    X, y = [], []
    for i in range(len(data) - hist_window - pred_window + 1):
        X.append(data[i:i+hist_window, 0])
        y.append(data[i+hist_window: i+hist_window+pred_window, 0])
    return np.array(X), np.array(y)


dates = pd.date_range(start="2020-04-12", end="2021-10-12")
dates = list(dates.strftime("%Y_%m_%d"))
raw_data = get_data(dates, "./data/", state="Alabama")
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(raw_data.reshape(-1, 1))

train_size = int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - train_size
train_data = data_scaled[:train_size, :]
test_data = data_scaled[train_size:, :]

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

train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

# calculate RMSE
train_rmse = np.sqrt(np.mean((train_predict - y_train)**2))
test_rmse = np.sqrt(np.mean((test_predict - y_test)**2))

print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)

predict = test_predict[:,-1]
sns.lineplot(x=[x for x in range(1,len(raw_data)+1)], y=raw_data)
sns.lineplot(x=[x+args.hist_window+train_size for x in range(len(predict))], y=predict)
plt.show()