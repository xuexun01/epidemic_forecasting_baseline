import torch
import torch.nn as nn
import torch.optim as optim


class SIRD(nn.Module):
    def __init__(self):
        super(SIRD, self).__init__()
    
    def forword(self, dS, dI, dR, dD):
        pass


class ARIMA(nn.Module):
    def __init__(self, p, d, q):
        super(ARIMA, self).__init__()
        self.p = p
        self.d = d
        self.q = q
        self.ar = nn.Linear(p, 1)
        self.ma = nn.Linear(q, 1)
        self.diff = None
    
    def forward(self, x):
        if self.diff is not None:
            x = self.difference(x)
        return self.ar(x) + self.ma(x)
    
    def difference(self, x):
        if self.diff is None:
            self.diff = torch.zeros_like(x)
        diffed = x - self.diff
        self.diff = x
        return diffed

# 示例用法
# 创建 ARIMA 模型
model = ARIMA(p=1, d=1, q=1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 准备数据
data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).unsqueeze(1)  # 示例输入数据

# 训练模型
for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

# 使用模型进行预测
predictions = model(data)
print('Predictions:', predictions.squeeze().detach().numpy())

