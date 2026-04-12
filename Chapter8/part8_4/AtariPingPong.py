import gymnasium as gym
import time
import ale_py

# render_mode="human" 会弹出一个实时窗口展示画面
# "ALE/Pong-v5" 是经典乒乓球游戏
gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="human")

state, _ = env.reset()
done = False

for _ in range(1000):
    # 随机选择一个动作
    action = env.action_space.sample()

    # 执行动作
    observation, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated
    if done:
        state, info = env.reset()

    # 控制一下循环速度，否则画面闪得太快
    time.sleep(0.01)

env.close()