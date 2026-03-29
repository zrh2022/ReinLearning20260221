import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from DQNAgent import DQNAgent
import torch

env = gym.make("CartPole-v1")
dqn_agent = DQNAgent()
# dqn_agent.init_plot()  # 初始化画布

total_steps = 0
episodes = 10000
for episode in range(episodes):
    state, info = env.reset()
    total_reward = 0
    total_loss = 0
    done = False

    while not done:
        total_steps += 1
        # 这里建议用 agent.get_action(state)，而不是随机，否则学不到东西
        # action = np.random.choice(0, 1)
        action = dqn_agent.get_action(torch.tensor(state, dtype=torch.float32))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 存入并更新
        loss = dqn_agent.update_qnet((state, action, reward, next_state, done))

        total_reward += reward
        state = next_state
        if loss is not None:
            total_loss += loss

        if total_steps % 500 == 0:
            dqn_agent.sync_qnet_target()

    # Episode 结束后的处理
    dqn_agent.reward_history.append((episode, total_reward))
    print(f'Episode: {episode}, Reward: {total_reward}, ')
    if total_loss is not None:
        print(f'Loss: {total_loss / total_steps:.6f}')

    # 关键：每隔几个 Episode 刷新一次图表，不要每个 Step 都刷，太慢
    # if episode % 50 == 0:
    #     dqn_agent.update_plot()
    #     plt.pause(0.1)  # 给 GUI 呼吸的时间

plt.ioff()
plt.show()  # 训练完不关闭，保持显示
env.close()