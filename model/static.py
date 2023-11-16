import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


class SIRD(nn.Module):
    def __init__(self):
        super(SIRD, self).__init__()

    def forword(self, dS, dI, dR, dD):
        pass


class ARIMA(nn.Module):
    def __init__(self, pred_window, hist_window, d=1, q=4):
        super(ARIMA, self).__init__()
        self.p = hist_window
        self.pred_window = pred_window
        self.d = d
        self.q = q
        self.ar = nn.Linear(hist_window, 1)
        self.ma = nn.Linear(q, 1)
        self.diff = None

    def forward(self, x):
        x = np.transpose(x, (0, 2, 1))
        if self.diff is not None:
            x = self.difference(x)
        return self.ar(x) + self.ma(x)

    def difference(self, x):
        if self.diff is None:
            self.diff = torch.zeros_like(x)
        diffed = x - self.diff
        self.diff = x
        return diffed


model = ARIMA(pred_window=12, hist_window=2, d=1, q=4)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

data = []

# 训练模型
for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# 使用模型进行预测
predictions = model(data)
print("Predictions:", predictions.squeeze().detach().numpy())