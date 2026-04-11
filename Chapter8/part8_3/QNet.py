import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # 输入：(batch_size, 4, 84, 84)
        self.cv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)  # 输出(batch_size, 32, 20, 20)
        self.cv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)  # 输出(batch_size, 64, 9, 9)
        self.cv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)  # 输出(batch_size, 64, 7, 7)
        self.flatten = nn.Flatten()  # 输出(batch_size, 3136)
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输出(batch_size, 100)
        self.fc2 = nn.Linear(hidden_size, output_size)   # 输出(batch_size, 6)

    def forward(self, x):
        x = F.relu(self.cv1(x))
        x = F.relu(self.cv2(x))
        x = F.relu(self.cv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
