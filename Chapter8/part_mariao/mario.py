import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import pygame

pygame.init()
screen = pygame.display.set_mode((256, 240))
pygame.display.set_caption("Super Mario Bros")

env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state = env.reset()
clock = pygame.time.Clock()

# SIMPLE_MOVEMENT 动作列表:
# 0: ['NOOP']
# 1: ['right']
# 2: ['right', 'A']       <- 向右跳跃
# 3: ['right', 'B']       <- 向右加速
# 4: ['right', 'A', 'B']  <- 向右加速跳跃
# 5: ['A']                <- 原地跳跃
# 6: ['left']

def get_action(keys):
    # 向右加速跳跃
    if keys[pygame.K_RIGHT] and keys[pygame.K_SPACE] and keys[pygame.K_LSHIFT]:
        return 4
    # 向右跳跃
    elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return 2
    # 向右加速
    elif keys[pygame.K_RIGHT] and keys[pygame.K_LSHIFT]:
        return 3
    # 向左
    elif keys[pygame.K_LEFT]:
        return 6
    # 原地跳跃
    elif keys[pygame.K_SPACE]:
        return 5
    # 向右
    elif keys[pygame.K_RIGHT]:
        return 1
    else:
        return 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            pygame.quit()
            exit()

    keys = pygame.key.get_pressed()
    action = get_action(keys)

    state, reward, done, info = env.step(action)

    # 将游戏画面渲染到 pygame 窗口
    frame = state  # shape: (240, 256, 3)
    surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    clock.tick(60)

    if done:
        state = env.reset()