from collections import deque
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from Chapter7.part7_4.QNet import QNet


# 改动：神经网络版本的Q学习
class Agent:
    def __init__(self, env, gamma=0.9, epsilon=0.1, alpha=0.1, action_size=4, hidden_size=100):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        self.alpha = alpha
        self.action_size = action_size

        self.qnet = QNet(input_size=len(env.getStates()), hidden_size=hidden_size, output_size=action_size)
        # 3. 定义损失函数和优化器
        self.criterion = nn.MSELoss()  # 均方误差损失
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=1e-4)
        
        # 用于线程间通信
        self.loss_history = []
        self.lock = threading.Lock()
        self.stop_plotting = False

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet.forward(self.one_hot_state(state))
            return torch.argmax(qs).item()

    def get_best_policy(self, env):
        pi = {}
        for state in env.getStates():
            qs = self.qnet.forward(self.one_hot_state(state))
            max_action = torch.argmax(qs).item()
            pi[state] = max_action
        return pi

    def get_Q(self):
        Q = {}
        for state in self.env.getStates():
            qs = self.qnet.forward(self.one_hot_state(state))
            for a in range(self.action_size):
                key = (state, a)
                Q[key] = qs[a].item()
        return Q

    def run(self, episodes):
        # 启动绘图线程
        # plot_thread = threading.Thread(target=self._plot_worker, daemon=True)
        # plot_thread.start()
        
        for episode in range(episodes):
            state = self.env.start_state
            total_loss = 0.0
            step_count = 0

            while True:
                action = self.get_action(state)
                next_state = self.env.getNextState(state, action)
                reward = self.env.getReward(next_state)
                done = self.env.get_is_done(next_state)
                loss = self.update_qnet(state, action, reward, next_state, done)
                total_loss += loss
                step_count += 1
                
                if done:
                    avg_loss = total_loss / step_count
                    
                    # 将 loss 添加到历史记录（线程安全）
                    with self.lock:
                        self.loss_history.append((episode, avg_loss))
                    
                    print(f'Episode: {episode}, Reward: {reward}, Avg Loss: {avg_loss:.6f}')
                    break
                state = next_state
        
        # 训练结束，等待最后一次更新
        # import time
        # time.sleep(0.5)
        # self.stop_plotting = True
        # plot_thread.join()

    def _plot_worker(self):
        """后台绘图线程"""
        plt.ion()  # 交互模式
        fig, ax = plt.subplots(figsize=(10, 6))
        
        while not self.stop_plotting:
            with self.lock:
                if len(self.loss_history) > 0:
                    episodes, losses = zip(*self.loss_history)
                    
                    ax.clear()
                    ax.plot(episodes, losses, label='Training Loss', linewidth=2, color='#1f77b4')
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Average Loss')
                    ax.set_title('Real-time Training Loss Curve')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            
            # 每 0.5 秒更新一次图表
            plt.pause(0.5)
        
        plt.close(fig)

    def one_hot_state(self, state):
        x, y = state
        return F.one_hot(torch.tensor(x * 4 + y), num_classes=12).float()

    def update_qnet(self, state, action, reward, next_state, is_done):
        # 1. 计算 Target
        with torch.no_grad():
            if is_done:
                target_value = torch.tensor(reward, dtype=torch.float32)
                cal_target = self.qnet(self.one_hot_state(state))
                print(f"target: {torch.max(cal_target).item()}")
            else:
                qs_next = self.qnet(self.one_hot_state(next_state))
                target_value = reward + self.gamma * torch.max(qs_next)

            # 在 update_qnet 计算 target 之后
            # target_value = torch.clamp(target_value, min=-1.0, max=1.0)

        # 更新 Q 值
        qs = self.qnet(self.one_hot_state(state))
        current_q = qs[action]  # 只取当前动作的Q值

        # 正确写法: 使用 F.mse_loss()
        loss = F.smooth_l1_loss(current_q, target_value)

        self.qnet.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
