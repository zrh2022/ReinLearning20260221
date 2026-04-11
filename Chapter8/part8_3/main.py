import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from DQNAgent import DQNAgent
import torch
import ale_py


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda use device: " + str(device))

# render_mode="human" 会弹出一个实时窗口展示画面
# "ALE/Pong-v5" 是经典乒乓球游戏
gym.register_envs(ale_py)
# env = gym.make("ALE/Pong-v5", render_mode="human")
env = gym.make("ALE/Pong-v5", render_mode=None, frameskip=4)

dqn_agent = DQNAgent(device=device)
# dqn_agent.init_plot()  # 初始化画布

total_steps = 0
episodes = 10000
action_repeat = 4
for episode in range(episodes):
    state, info = env.reset()
    total_reward = 0
    total_loss = 0
    done = False
    action = None  # 保存上一次动作
    processed_image = None
    loss = 0.0
    env.step(1)  # FIRE

    # 动作选择 Epsilon 衰减
    if episode % 50 == 0:
        dqn_agent.epsilon = max(dqn_agent.epsilon_min, dqn_agent.epsilon * dqn_agent.epsilon_decay)

    while not done:
        total_steps += 1

        # 预处理图像
        if processed_image is None:
            processed_image = dqn_agent.image_pre_process(state)

        # 每 ACTION_REPEAT 步执行一次新动作
        if action is None or total_steps % action_repeat == 1:
            action = dqn_agent.get_action(processed_image.unsqueeze(0))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # 奖励裁剪：-1, 0, 1
        if reward == 0:
            clipped_reward = 0.01
        else:
            clipped_reward = np.sign(reward)  # (-1, 0 , 1)

        processed_next_image = dqn_agent.image_pre_process(next_state)
        # 存入并更新
        if total_steps % 4 == 0:
            loss = dqn_agent.update_qnet((processed_image, action, clipped_reward, processed_next_image, done))

        total_reward += reward
        state = next_state
        processed_image = processed_next_image
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
    # if episode % 10 == 0:
    #     dqn_agent.update_plot()
    #     plt.pause(0.01)  # 给 GUI 呼吸的时间

plt.ioff()
plt.show()  # 训练完不关闭，保持显示
env.close()
