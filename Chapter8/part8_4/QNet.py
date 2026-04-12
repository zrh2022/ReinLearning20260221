import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # 输入：(batch_size, 4, 84, 84)
        self.cv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)  # 输出(batch_size, 32, 20, 20)
        self.cv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)  # 输出(batch_size, 64, 9, 9)
        self.cv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)  # 输出(batch_size, 64, 7, 7)
        self.flatten = nn.Flatten()  # 输出(batch_size, 3136)

        # Dueling 网络分支
        # 价值函数，输出(batch_size, 1)
        self.value_stream = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # 优势函数，输出(batch_size, 6)
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = F.relu(self.cv1(x))
        x = F.relu(self.cv2(x))
        x = F.relu(self.cv3(x))
        x = self.flatten(x)

        # Dueling 分支
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Q(s, a) = V(s) + (A(s, a) - mean(A))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q
