import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Buffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, data):
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        # 从 data 中提取状态、动作、奖励、下一状态和终止标志，并转换为 PyTorch 张量
        states = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)  # 将每个 state 转换为张量，并堆叠
        actions = torch.tensor(np.array([d[1] for d in data]), dtype=torch.long)  # 动作通常是离散的，使用 long 类型
        rewards = torch.tensor(np.array([d[2] for d in data]), dtype=torch.float32)  # 奖励通常是浮点数
        next_states = torch.tensor(np.array([d[3] for d in data]), dtype=torch.float32)  # 将每个 next_state 转换为张量，并堆叠
        is_done = torch.tensor(np.array([d[4] for d in data]), dtype=torch.int)  # done 是布尔值
        return states, actions, rewards, next_states, is_done
