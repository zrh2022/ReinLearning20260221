import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class GridWorld:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # self.reward_map = np.array(
        #     [[0, 0, 0, 1.0],
        #      [0, None, 0, -1],
        #      [0, 0, 0, 0]]
        # )
        self._generate_random_reward_map()
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: '上',
            1: '下',
            2: '左',
            3: '右'
        }

        # self.goal_state = (0, 3)
        # self.start_state = (2, 0)
        # self.wall_state = [(1, 1)]
        # self.boom_state = (1, 3)
        self.agent_state = self.start_state

    def _generate_random_reward_map(self):
        """随机生成奖励地图"""
        # 初始化奖励地图，所有位置默认为 0
        self.reward_map = np.zeros((self.height, self.width))

        # 随机选择墙的位置（至少保留一个位置不为墙）
        total_cells = self.width * self.height
        num_walls = np.random.randint(0, max(1, total_cells // 5))  # 墙的数量不超过总格子的 1/5

        all_positions = [(i, j) for i in range(self.height) for j in range(self.width)]
        wall_positions = []

        if num_walls > 0:
            wall_indices = np.random.choice(len(all_positions), size=min(num_walls, len(all_positions) - 1),
                                            replace=False)
            wall_positions = [all_positions[idx] for idx in wall_indices]
            self.wall_state = wall_positions

        # 将墙的位置设为 None
        for pos in wall_positions:
            self.reward_map[pos] = None

        # 从非墙位置中选择炸弹和终点
        available_positions = [pos for pos in all_positions if pos not in wall_positions]

        # 随机选择终点位置（奖励为 1）
        goal_idx = np.random.randint(0, len(available_positions))
        self.goal_state = available_positions[goal_idx]
        self.reward_map[self.goal_state] = 1.0

        # 从剩余位置中随机选择炸弹位置（奖励为 -1）
        remaining_positions = [pos for pos in available_positions if pos != self.goal_state]

        # 随机生成多个炸弹
        num_booms = np.random.randint(1, max(1, len(remaining_positions) + 1))  # 至少 1 个炸弹
        num_booms = min(num_booms, total_cells // 3)  # 最多 5 个炸弹
        if len(remaining_positions) > 0:
            # 确保炸弹数量不超过可用位置数量
            num_booms = min(num_booms, len(remaining_positions))
            # 随机选择多个炸弹位置（不重复）
            boom_indices = np.random.choice(len(remaining_positions), size=num_booms, replace=False)
            self.boom_state = [remaining_positions[idx] for idx in boom_indices]
            for boom_pos in self.boom_state:
                self.reward_map[boom_pos] = -1.0
        else:
            # 如果没有剩余位置，就不设置炸弹
            self.boom_state = []

        # 随机选择起点（不能是墙、炸弹或终点）
        start_positions = [pos for pos in remaining_positions if pos != self.boom_state]
        if len(start_positions) > 0:
            start_idx = np.random.randint(0, len(start_positions))
            self.start_state = start_positions[start_idx]
        else:
            # 默认起点在左下角
            self.start_state = (self.height - 1, 0)

        self.agent_state = self.start_state

    def getStates(self):
        return [(i, j) for i in range(self.height) for j in range(self.width)]

    def getReward(self, state):
        return self.reward_map[state]

    def getNextState(self, state, action):
        next_state = state
        if action == 0 and state[0] - 1 >= 0:
            next_state = state[0] - 1, state[1]
        elif action == 1 and state[0] + 1 < self.height:
            next_state = state[0] + 1, state[1]
        elif action == 2 and state[1] - 1 >= 0:
            next_state = state[0], state[1] - 1
        elif action == 3 and state[1] + 1 < self.width:
            next_state = state[0], state[1] + 1

        if next_state in self.wall_state:
            next_state = state
        return next_state

    def render(self, V=None, policy=None):
        # 根据网格大小动态调整图形尺寸
        figsize_width = max(8, self.width * 2)
        figsize_height = max(6, self.height * 2)
        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

        # 根据 V 值创建颜色网格
        v_grid = np.zeros((self.height, self.width))
        wall_mask = np.zeros((self.height, self.width), dtype=bool)

        # 填充 V 值网格
        for i in range(self.height):
            for j in range(self.width):
                state = (i, j)
                if state in self.wall_state:
                    wall_mask[i, j] = True
                    v_grid[i, j] = 0  # 墙设为 0（白色）
                elif V is not None and state in V:
                    v_grid[i, j] = V[state]
                else:
                    v_grid[i, j] = 0

        # 创建自定义 colormap: -1(绿) -> 0(白) -> 1(红)
        colors = [(1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0)]
        cmap = LinearSegmentedColormap.from_list('V_Colormap', colors, N=256)

        # 绘制 V 值热力图，墙区域用灰色覆盖
        im = ax.imshow(v_grid, cmap=cmap, vmin=-1, vmax=1, alpha=0.7)

        # 为墙添加灰色遮罩
        if wall_mask.any():
            gray_mask = np.ma.masked_where(~wall_mask, wall_mask)
            ax.imshow(gray_mask, cmap=ListedColormap(['gray']), alpha=0.5, vmin=0, vmax=1)

        # 画直线
        ax.set_xticks(np.arange(-0.5, self.width, 1))
        ax.set_yticks(np.arange(-0.5, self.height, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='both', color='black', linewidth=2)

        # Add text annotations for each cell
        for i in range(self.height):
            for j in range(self.width):
                state = (i, j)

                # Skip wall rendering
                if state in self.wall_state:
                    ax.text(j, i, '墙\n障', ha='center', va='center',
                            fontsize=20, fontweight='bold', color='white')
                    continue

                # Add reward information
                reward = self.reward_map[i, j]
                if reward != 0:
                    ax.text(j, i - 0.3, f'R={reward}', ha='center', va='center',
                            fontsize=18, fontweight='bold', color='darkgreen')

                # Add state value if provided
                if V is not None and state in V:
                    # 根据 V 值决定文字颜色，确保在彩色背景上清晰可见
                    v_value = V[state]
                    ax.text(j, i + 0.35, f'V={v_value:.2f}', ha='center',
                            va='center', fontsize=11, fontweight='bold',
                            color='black')

                # Add policy arrow if provided
                if policy is not None and state in policy:
                    action_probs = policy[state]
                    best_action = max(action_probs, key=action_probs.get)
                    arrow_dir = {'上': (0, -0.3), '下': (0, 0.3),
                                 '左': (-0.3, 0), '右': (0.3, 0)}
                    direction = self.action_meaning[best_action]
                    dx, dy = arrow_dir[direction]
                    ax.arrow(j, i, dx * 0.4, dy * 0.4, head_width=0.2,
                             head_length=0.15, fc='black', ec='black',
                             linewidth=1)

        # Mark agent position
        ax.plot(self.agent_state[1], self.agent_state[0], 'o',
                markersize=15, markeredgecolor='purple',
                markerfacecolor='yellow', label='Agent')

        # Mark start and goal positions
        ax.plot(self.start_state[1], self.start_state[0], 's',
                markersize=12, markeredgecolor='black',
                markerfacecolor='lightblue', label='Start')
        ax.plot(self.goal_state[1], self.goal_state[0], '*',
                markersize=20, markeredgecolor='gold',
                markerfacecolor='green', label='Goal')

        # Mark boom positions with bomb shape (if exists)
        if self.boom_state is not None and len(self.boom_state) > 0:
            for idx, boom_pos in enumerate(self.boom_state):
                # Draw bomb body (black circle)
                bomb_body = plt.Circle((boom_pos[1], boom_pos[0]),
                                       0.25, facecolor='none', edgecolor='black', linewidth=2,
                                       label='Boom' if idx == 0 else "")
                ax.add_patch(bomb_body)

                # Draw fuse (brown line on top)
                ax.plot([boom_pos[1] + 0.05, boom_pos[1] + 0.25],
                        [boom_pos[0] - 0.05, boom_pos[0] - 0.25],
                        color='#8B4513', linewidth=3, solid_capstyle='round')

                # Draw spark at the end of fuse (red/orange circle)
                spark = plt.Circle((boom_pos[1] + 0.25, boom_pos[0] - 0.25),
                                   0.06, facecolor='#FF4500', edgecolor='none')
                ax.add_patch(spark)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('状态价值 V', rotation=270, labelpad=15, fontsize=12)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

        # Add legend and title
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        ax.set_title('Grid World - 状态价值可视化', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()

# grid_world = GridWorld(4, 3)
# for cur_state in grid_world.getStates():
#     print(cur_state)
#
# # 随机生成V
# V = {state: np.random.uniform(-1, 1) for state in grid_world.getStates()}
# grid_world.render(V=V)
