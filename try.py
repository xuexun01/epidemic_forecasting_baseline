import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def get_data(dates, data_path, state):
    cases = []
    for date in dates:
        df = pd.read_csv(f"{data_path}covid_19_daily_reports_us/{date}.csv")
        cases.append(
            df.loc[df["Province_State"] == state, "Confirmed"].values[0]
        )
    return np.array(cases)

dates = pd.date_range(start="2020-04-12", end="2021-10-12")
dates = list(dates.strftime("%Y_%m_%d"))
raw_data = get_data(dates, "./data/", state="Alabama")

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(raw_data.reshape(-1, 1))

# 创建训练集和测试集
train_size = int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - train_size
train_data = data_scaled[:train_size, :]
test_data = data_scaled[train_size:, :]

# 创建时间窗口数据
def create_time_series(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0])
        y.append(data[i+window_size, 0])
    return np.array(X), np.array(y)

window_size = 10
X_train, y_train = create_time_series(train_data, window_size)
X_test, y_test = create_time_series(test_data, window_size)
print(X_train.shape)
print(y_train.shape)

# 将输入数据重塑为LSTM所需的张量形状
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 模型训练
model.fit(X_train, y_train, epochs=100, batch_size=32)

# 模型预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反标准化
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# 计算RMSE指标
train_rmse = np.sqrt(np.mean((train_predict - y_train)**2))
test_rmse = np.sqrt(np.mean((test_predict - y_test)**2))

print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)

print(y_test)
print(test_predict)