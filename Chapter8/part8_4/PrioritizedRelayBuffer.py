import numpy as np
import torch


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size=32, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree(buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.batch_size = batch_size
        self.epsilon = 1e-6  # 防止优先级为 0
        self.max_priority = 1.0  # 初始最大优先级设为 1.0

    def add(self, sample):
        # 使用记录的最大优先级 p 存入新数据，确保新数据至少被采样一次
        self.tree.add(self.max_priority, sample)

    def __len__(self):
        return self.tree.n_entries

    # 修改 get_batch 方法中的权重计算部分
    def get_batch(self):
        if self.tree.n_entries == 0:
            return [], [], []

        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        weights = np.zeros((self.batch_size,))

        # 找到当前 tree 中最小的概率（用于归一化权重，提高稳定性）
        # 如果 tree 还没满，最小概率通常就是 self.epsilon ** self.alpha
        prob_min = (self.epsilon ** self.alpha) / self.tree.total()

        for i in range(self.batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)

            # 防止 p 为 0 导致除以零报错
            p_safe = max(p, self.epsilon)
            batch.append(data)
            idxs.append(idx)

            # 概率 P(i) = p_i / total_p
            probability = p_safe / self.tree.total()
            # 权重计算公式: w_i = (N * P(i)) ^ -beta
            weights[i] = (self.tree.n_entries * probability) ** (-self.beta)

        # 归一化权重，确保所有权重 <= 1，防止梯度爆炸
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        return idxs, batch, weights

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            # 更新记录的最大优先级
            self.max_priority = max(self.max_priority, p)
