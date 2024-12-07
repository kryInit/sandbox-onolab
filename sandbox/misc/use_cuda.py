import time

import torch
import torch.nn as nn
import torch.optim as optim


# 大きめのニューラルネットワークの定義
class LargeNN(nn.Module):
    def __init__(self):
        super(LargeNN, self).__init__()
        self.fc1 = nn.Linear(10000, 5000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(5000, 1000)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1000, 100)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


# 学習関数
def train(model, device, optimizer, criterion, data, target):
    model.train()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()


# 実行時間を測定する関数
def measure_time(device):
    model = LargeNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # ダミーデータ生成
    data = torch.randn(1024, 10000)  # バッチサイズを1024に増やし、入力データを大きくします
    target = torch.randint(0, 10, (1024,))

    # 実行時間計測
    start_time = time.time()
    for _ in range(100):  # エポック数を100に増やして計算量を増やします
        train(model, device, optimizer, criterion, data, target)
    end_time = time.time()

    return end_time - start_time


# CPUとGPUで実行時間を比較
cpu_time = measure_time(torch.device("cpu"))
print(f"CPU time: {cpu_time:.2f} seconds")

if torch.cuda.is_available():
    gpu_time = measure_time(torch.device("cuda"))
    print(f"GPU time: {gpu_time:.2f} seconds")
else:
    print("GPU is not available.")
