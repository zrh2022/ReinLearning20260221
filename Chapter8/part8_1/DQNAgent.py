import threading

import numpy as np
from matplotlib import pyplot as plt

from Chapter8.part8_1.Buffer import Buffer
from Chapter8.part8_1.QNet import QNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQNAgent:
    def __init__(self, gamma=0.9, epsilon=0.1, input_size=4, action_size=2, hidden_size=100,
                 buffer_size=10000, batch_size=32):

        self.epsilon = epsilon
        self.gamma = gamma
        self.action_size = action_size
        # 主网络
        self.qnet = QNet(input_size=input_size, hidden_size=hidden_size, output_size=action_size)
        # 目标网络
        self.qnet_target = QNet(input_size=input_size, hidden_size=hidden_size, output_size=action_size)
        self.buffer = Buffer(buffer_size=buffer_size, batch_size=batch_size)

        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()  # 均方误差损失
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=1e-4)

        # 用于线程间通信
        self.reward_history = []
        self.lock = threading.Lock()
        self.stop_plotting = False

    def sync_qnet_target(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet.forward(state)
            return torch.argmax(qs).item()

    def update_qnet(self, data):
        # 加入单个数据
        self.buffer.add(data)
        if len(self.buffer) < self.buffer.batch_size:
            return

        # 经验回放：取出批量数据
        state, action, reward, next_state, is_done = self.buffer.get_batch()
        # 1. 确保维度统一 (把 shape 从 [32] 变成 [32, 1])
        reward = reward.unsqueeze(1)
        is_done = is_done.unsqueeze(1).float()  # 确保是 float 才能相乘
        with torch.no_grad():
            # 2. 计算下一状态的所有动作 Q 值 (32, 2)
            qs_next = self.qnet_target.forward(next_state)
            # 3. 找到每个样本的最大 Q 值 (32, 1)
            # torch.max(input, dim) 返回 (values, indices)，我们只需要 values
            max_qs_next, _ = torch.max(qs_next, dim=1, keepdim=True)

            # 4. 贝尔曼方程批量计算 (所有运算现在都是 [32, 1] vs [32, 1])
            # 如果 is_done 为 1，则后面那一项变为 0
            target_value = reward + self.gamma * (1 - is_done) * max_qs_next

        # 5. 计算当前 Q 值
        # action 的 shape 也应该是 (32, 1)
        all_qs = self.qnet(state)  # (32, 2)
        # 使用 gather 选出实际执行的那个 action 对应的 Q 值
        current_q = all_qs.gather(1, action.unsqueeze(1))  # 结果为 (32, 1)

        # 正确写法: 使用 F.mse_loss()
        loss = F.smooth_l1_loss(current_q, target_value)

        self.qnet.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # DQNAgent 内部
    def init_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line, = self.ax.plot([], [], label='Training Reward', color='#1f77b4')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.ax.legend()

    def update_plot(self):
        if len(self.reward_history) > 0:
            episodes, rewards = zip(*self.reward_history)
            self.line.set_data(episodes, rewards)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

