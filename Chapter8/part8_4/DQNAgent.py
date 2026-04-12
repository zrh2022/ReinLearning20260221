import os
import threading

import numpy as np
from matplotlib import pyplot as plt
import cv2

# from Chapter8.part8_4.Buffer import Buffer
from Chapter8.part8_4.PrioritizedRelayBuffer import PrioritizedReplayBuffer  # 修改导入
from Chapter8.part8_4.QNet import QNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQNAgent:
    def __init__(self, gamma=0.99, epsilon=0.1, input_size=3136, action_size=6, hidden_size=512,
                 buffer_size=50000, batch_size=32, cached_frames=4, device="cpu"):

        self.device = device
        # self.epsilon = epsilon
        self.gamma = gamma
        self.action_size = action_size
        self.cached_frames = cached_frames

        # 修改 epsilon 自衰减
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # 新增：缓存最近 4 帧
        self.frame_stack = []

        # 主网络
        self.qnet = QNet(input_size=input_size, hidden_size=hidden_size, output_size=action_size).to(device)
        # 目标网络
        self.qnet_target = QNet(input_size=input_size, hidden_size=hidden_size, output_size=action_size).to(device)
        # self.buffer = Buffer(buffer_size=buffer_size, batch_size=batch_size)
        # 更换为优先级缓存
        self.buffer = PrioritizedReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()  # 均方误差损失
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=5e-5)

        # 用于线程间通信
        self.reward_history = []
        self.lock = threading.Lock()
        self.stop_plotting = False

    # def image_pre_process(self, image):
    #     # 1. 灰度化
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     # 2. 缩放至 84x84 像素
    #     image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
    #     # 3. 归一化到 [0, 1]
    #     image = image / 255.0
    #     # 4. 增加通道维度 (84, 84) -> (1, 84, 84)
    #     image = np.expand_dims(image, axis=0)
    #     image_tensor = torch.tensor(image, dtype=torch.float32, device=self.device)
    #
    #     # 堆叠帧
    #     if len(self.frame_stack) == 0:
    #         self.frame_stack = [image_tensor] * self.cached_frames  # 初始化为 4 帧
    #     else:
    #         self.frame_stack.pop(0)
    #         self.frame_stack.append(image_tensor)
    #
    #     # 堆叠为 (4, 84, 84)
    #     stacked_frames = torch.cat(self.frame_stack, dim=0)  # shape: (4, 84, 84)
    #     return stacked_frames

    def image_pre_process(self, image):
        with torch.no_grad():  # ← 整个预处理不需要梯度，加这个
            image = image[34:194, :, :]
            image_tensor = torch.tensor(image, dtype=torch.float32, device=self.device)
            # RGB -> Gray
            image_tensor = image_tensor.mean(dim=2, keepdim=True)
            # resize
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1,1,H,W)
            image_tensor = F.interpolate(image_tensor, size=(84, 84), mode='area')

            image_tensor = image_tensor / 255.0
            image_tensor = image_tensor.squeeze(0)  # (1,84,84)

            if len(self.frame_stack) == 0:
                self.frame_stack = image_tensor.repeat(self.cached_frames, 1, 1)
            else:
                self.frame_stack = torch.roll(self.frame_stack, shifts=-1, dims=0)
                self.frame_stack[-1] = image_tensor
            return self.frame_stack.clone()

    def sync_qnet_target(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def get_action(self, state):
        valid_actions = [0, 2, 3]
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                qs = self.qnet(state)
                # 只选 valid actions
                qs_valid = qs[:, valid_actions]
                best_idx = torch.argmax(qs_valid).item()

            return valid_actions[best_idx]

    def get_action_inference(self, state):
        valid_actions = [0, 2, 3]
        with torch.no_grad():
            qs = self.qnet_target(state)
            # 只选 valid actions
            qs_valid = qs[:, valid_actions]
            best_idx = torch.argmax(qs_valid).item()
        return valid_actions[best_idx]

    def update_qnet(self, data):
        # 加入单个数据
        self.buffer.add(data)
        if len(self.buffer) < self.buffer.batch_size:
            return

        # 经验回放：取出批量数据
        state, action, reward, next_state, is_done = self.buffer.get_batch()

        # 确保维度统一 (把 shape 从 [32] 变成 [32, 1])
        state = state.to(self.device)
        action = action.to(self.device).unsqueeze(1)
        reward = reward.to(self.device).unsqueeze(1)
        next_state = next_state.to(self.device)
        is_done = is_done.to(self.device).unsqueeze(1).float()

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
        current_q = all_qs.gather(1, action)  # 结果为 (32, 1)

        # 正确写法: 使用 F.mse_loss()
        loss = F.smooth_l1_loss(current_q, target_value)

        self.qnet.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # DQNAgent 里增加这个方法（原 update_qnet 改为只从 buffer 训练）
    def update_qnet_from_buffer(self):
        """只从 buffer 采样训练，数据添加由外部负责。"""
        if len(self.buffer) < self.buffer.batch_size:
            return None

        # 获取优先级采样
        idxs, batch, weights = self.buffer.get_batch()
        # 检查 batch 中每个元素是否可迭代
        if any(not isinstance(b, (tuple, list)) for b in batch):
            return None

        state, action, reward, next_state, is_done = zip(*batch)
        # state, action, reward, next_state, is_done = self.buffer.get_batch()

        # 转换为 Tensor 并移动到 device
        state = torch.stack(state).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_state = torch.stack(next_state).to(self.device)
        is_done = torch.tensor(is_done, dtype=torch.int).to(self.device).unsqueeze(1).float()

        with torch.no_grad():
            # Double DQN
            qs_next_target = self.qnet_target(next_state)
            qs_next = self.qnet(next_state)

            # 筛选有效动作
            valid_actions = [0, 2, 3]
            qs_next_valid = qs_next[:, valid_actions]

            # 找到最大 Q 值及其对应的动作索引
            max_qs_next, action_qs_next = torch.max(qs_next_valid, dim=1, keepdim=True)

            # 从目标网络中提取对应动作的 Q 值
            action_qs_next = torch.tensor([valid_actions[i] for i in action_qs_next[:, 0].tolist()],
                                          device=qs_next.device).unsqueeze(1)
            max_qs_next_target = qs_next_target.gather(1, action_qs_next)

            target_value = reward + self.gamma * (1 - is_done) * max_qs_next_target

        all_qs = self.qnet(state)
        current_q = all_qs.gather(1, action)

        # loss = F.smooth_l1_loss(current_q, target_value)
        # 优先级经验回放：使用权重调整损失
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device).unsqueeze(1)
        loss = (F.smooth_l1_loss(current_q, target_value, reduction='none') * weights).mean()

        self.optimizer.zero_grad()  # ✅ 用 optimizer.zero_grad() 而非 qnet.zero_grad()
        loss.backward()
        # ✅ 加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 更新优先级
        errors = (target_value - current_q).abs().cpu().detach().numpy()
        self.buffer.update_priorities(idxs, errors)

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

    # ===== 保存 =====
    def save_checkpoint(self, episode, total_steps, path="checkpoints"):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "episode": episode,
            "total_steps": total_steps,
            "qnet": self.qnet.state_dict(),
            "qnet_target": self.qnet_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "reward_history": self.reward_history,
        }, f"{path}/ckpt_ep{episode}.pth")
        print(f"  [Saved] checkpoint at episode {episode}")

    # ===== 加载 =====
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.qnet.load_state_dict(ckpt["qnet"])
        self.qnet_target.load_state_dict(ckpt["qnet_target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.reward_history = ckpt["reward_history"]
        print(f"  [Loaded] episode={ckpt['episode']}, steps={ckpt['total_steps']}, epsilon={ckpt['epsilon']:.4f}")
        return ckpt["episode"] + 1, ckpt["total_steps"]  # 返回下一个 episode 和已有 steps

