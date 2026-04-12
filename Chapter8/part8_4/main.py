import gc
import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from DQNAgent import DQNAgent
import torch
import ale_py


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device: " + str(device))

# render_mode="human" 会弹出一个实时窗口展示画面
# "ALE/Pong-v5" 是经典乒乓球游戏
gym.register_envs(ale_py)
# env = gym.make("ALE/Pong-v5", render_mode="human")
env = gym.make("ALE/Pong-v5", render_mode=None, frameskip=4)

# ✅ 修复⑤：用 frameskip=1 + repeat_action_probability=0 更可控，
#   或保持 frameskip=4 不变，但需要在失分后重新 FIRE（见下方循环）
dqn_agent = DQNAgent(device=device)
# dqn_agent.init_plot()  # 初始化画布

# ===== 训练开始前：尝试加载断点 =====
CHECKPOINT_DIR = "checkpoints"
RESUME_PATH = "checkpoints/ckpt_ep1000.pth"  # 改成具体路径即可续训，如 "checkpoints/ckpt_ep200.pth"

# ✅ 增加 Buffer 预热阈值
WARMUP_STEPS = 10000   # 至少收集这么多步才开始训练

# 确定多少步数同步目标网络
TARGET_SYNC_STEPS_START = 5000
TARGET_SYNC_STEPS_MAX = 15000
TARGET_SYNC_STEPS_EPOCH_STEPS = 10
TARGET_SYNC_STEPS = TARGET_SYNC_STEPS_START

total_steps = 0
episodes = 10000
action_repeat = 4

start_episode = 0
if RESUME_PATH and os.path.exists(RESUME_PATH):
    start_episode, total_steps = dqn_agent.load_checkpoint(RESUME_PATH)

for episode in range(start_episode, episodes):
    state, info = env.reset()
    total_reward = 0
    total_loss = 0
    loss_count = 0
    done = False
    action = None  # 保存上一次动作
    processed_image = None
    loss = 0.0
    # TARGET_SYNC_STEPS = min(TARGET_SYNC_STEPS_START + TARGET_SYNC_STEPS_EPOCH_STEPS * episode, TARGET_SYNC_STEPS_MAX)
    TARGET_SYNC_STEPS = 10000

    # ✅ 修复⑤：封装一个辅助函数，失分后重新发球
    # Pong 在每次失分后需要再次 FIRE（action=1）才会重新发球
    # info['lives'] 可以检测是否失分
    lives = info.get('lives', 0)
    env.step(1)  # 初始 FIRE

    while not done:
        total_steps += 1

        if processed_image is None:
            processed_image = dqn_agent.image_pre_process(state)

        # 每 ACTION_REPEAT 步执行一次新动作
        # if action is None or total_steps % action_repeat == 1:
        action = dqn_agent.get_action(processed_image.unsqueeze(0))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ✅ 修复⑤：失分后重新 FIRE
        new_lives = info.get('lives', 0)
        if new_lives < lives:
            env.step(1)  # 重新发球
            lives = new_lives

        # ✅ 修复③：奖励裁剪，0 就是 0，不要给 0.01
        clipped_reward = float(np.sign(reward))  # -1.0, 0.0, 1.0

        processed_next_image = dqn_agent.image_pre_process(next_state)

        # ✅ 修复④：只有超过预热阈值才开始训练
        if total_steps % 4 == 0:
            # 先把数据加入 buffer（无论是否开始训练）
            dqn_agent.buffer.add(
                (processed_image, action, clipped_reward, processed_next_image, done)
            )
            if total_steps >= WARMUP_STEPS:
                loss = dqn_agent.update_qnet_from_buffer()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

        total_reward += reward
        state = next_state
        processed_image = processed_next_image

        # ✅ 修复②：目标网络同步改为 1000 步
        if total_steps % TARGET_SYNC_STEPS == 0:
            dqn_agent.sync_qnet_target()
            print(f"  [Step {total_steps}] Target network synced")

        # 每 1000 步清理一次显存碎片
        if total_steps % 1000 == 0:
            torch.cuda.empty_cache()

    # ✅ 修复①：Epsilon 衰减移到 episode 结束后，每 episode 都衰减
    dqn_agent.epsilon = max(
        dqn_agent.epsilon_min,
        dqn_agent.epsilon * dqn_agent.epsilon_decay
    )

    dqn_agent.reward_history.append((episode, float(total_reward)))  # 加 float()
    avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
    print(f"Episode: {episode:4d} | Reward: {total_reward:6.1f} | TARGET_SYNC_STEPS: {TARGET_SYNC_STEPS} | "
          f"Epsilon: {dqn_agent.epsilon:.4f} | Loss: {avg_loss:.6f} | Steps: {total_steps}")

    # 关键：每隔几个 Episode 刷新一次图表，不要每个 Step 都刷，太慢
    # if episode % 50 == 0:
    #     dqn_agent.update_plot()
    #     plt.pause(0.01)  # 给 GUI 呼吸的时间

    # ===== 每 100 个 episode 保存一次 =====
    if episode % 100 == 0:
        dqn_agent.save_checkpoint(episode, total_steps, CHECKPOINT_DIR)

    # 在 episode 循环结束处
    if episode % 10 == 0:
        gc.collect()  # 清理内存垃圾
        torch.cuda.empty_cache()  # 清理显存碎片

# ===== 训练结束后保存最终权重 =====
dqn_agent.save_checkpoint(episodes - 1, total_steps, CHECKPOINT_DIR)

plt.ioff()
plt.show()  # 训练完不关闭，保持显示
env.close()
