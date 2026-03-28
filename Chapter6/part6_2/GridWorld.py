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
        self.reward_map = np.array(
            [[0, 0, 0, 1.0],
             [0, None, 0, -1],
             [0, 0, 0, 0]]
        )
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: '上',
            1: '下',
            2: '左',
            3: '右'
        }

        self.goal_state = (0, 3)
        self.start_state = (2, 0)
        self.wall_state = (1, 1)
        self.boom_state = (1, 3)
        self.agent_state = self.start_state

    def getStates(self):
        return [(i, j) for i in range(self.height) for j in range(self.width)]

    def getReward(self, state):
        return self.reward_map[state]

    def get_is_done(self, state):
        return state == self.goal_state

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

        if next_state == self.wall_state:
            next_state = state
        return next_state

    def render_V(self, V=None, policy=None):
        fig, ax = plt.subplots(figsize=(8, 6))

        # 根据 V 值创建颜色网格
        v_grid = np.zeros((self.height, self.width))
        wall_mask = np.zeros((self.height, self.width), dtype=bool)

        # 填充 V 值网格
        for i in range(self.height):
            for j in range(self.width):
                state = (i, j)
                if state == self.wall_state:
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
                if state == self.wall_state:
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

        # Mark boom position with bomb shape
        # Draw bomb body (black circle)
        bomb_body = plt.Circle((self.boom_state[1], self.boom_state[0]),
                               0.25, facecolor='none', edgecolor='black', linewidth=2,
                               label='Boom')
        ax.add_patch(bomb_body)

        # Draw fuse (brown line on top)
        ax.plot([self.boom_state[1] + 0.05, self.boom_state[1] + 0.25],
                [self.boom_state[0] - 0.05, self.boom_state[0] - 0.25],
                color='#8B4513', linewidth=3, solid_capstyle='round')

        # Draw spark at the end of fuse (red/orange circle)
        spark = plt.Circle((self.boom_state[1] + 0.25, self.boom_state[0] - 0.25),
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

    def render_Q(self, Q=None):
        """
        可视化 Q 函数 - 在每个格子内用箭头展示四个动作的 Q 值
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # 创建基础网格和墙遮罩
        v_grid = np.zeros((self.height, self.width))
        wall_mask = np.zeros((self.height, self.width), dtype=bool)

        for i in range(self.height):
            for j in range(self.width):
                state = (i, j)
                if state == self.wall_state:
                    wall_mask[i, j] = True
                    v_grid[i, j] = 0
                else:
                    v_grid[i, j] = 0.5

        # 绘制背景网格
        cmap = plt.get_cmap('Greys')
        im = ax.imshow(v_grid, cmap=cmap, vmin=0, vmax=1, alpha=0.3)

        # 为墙添加灰色遮罩
        if wall_mask.any():
            gray_mask = np.ma.masked_where(~wall_mask, wall_mask)
            ax.imshow(gray_mask, cmap=ListedColormap(['gray']), alpha=0.5)

        # 画网格线
        ax.set_xticks(np.arange(-0.5, self.width, 1))
        ax.set_yticks(np.arange(-0.5, self.height, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='both', color='black', linewidth=2)

        # 定义箭头方向和样式
        arrow_configs = {
            0: {'label': '上', 'dx': 0, 'dy': -0.35, 'color': 'blue'},  # 上 - 蓝色
            1: {'label': '下', 'dx': 0, 'dy': 0.35, 'color': 'green'},  # 下 - 绿色
            2: {'label': '左', 'dx': -0.35, 'dy': 0, 'color': 'orange'},  # 左 - 橙色
            3: {'label': '右', 'dx': 0.35, 'dy': 0, 'color': 'red'}  # 右 - 红色
        }

        # 添加 Q 值和箭头标注
        for i in range(self.height):
            for j in range(self.width):
                state = (i, j)

                # 跳过墙障
                if state == self.wall_state:
                    ax.text(j, i, '墙\n障', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='white')
                    continue

                # 获取四个动作的 Q 值
                q_values = [Q.get((state, action_idx), 0.0) for action_idx in range(4)]
                max_q = max(q_values)
                min_q = min(q_values)
                q_range = max_q - min_q if max_q != min_q else 1.0

                # 在格子中心显示平均 Q 值或最大 Q 值
                avg_q = np.mean(q_values)
                best_action = q_values.index(max_q)

                ax.text(j, i, f'{avg_q:.2f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                # 为每个动作绘制箭头
                if Q is not None:
                    for action_idx in range(4):
                        q_value = Q.get((state, action_idx), 0.0)

                        # 根据 Q 值归一化箭头长度和宽度
                        normalized_q = (q_value - min_q) / q_range if q_range > 0 else 0.5
                        arrow_length = 0.15 + normalized_q * 0.15  # 箭头长度：0.15-0.3
                        arrow_width = 0.5 + normalized_q * 1.5  # 箭头宽度：0.5-2.0

                        config = arrow_configs[action_idx]

                        # 只绘制 Q 值大于最小值的箭头（避免太多箭头）
                        # if q_value > min_q or action_idx == best_action:
                        ax.annotate('',
                                    xy=(j + config['dx'], i + config['dy']),
                                    xytext=(j, i),
                                    arrowprops=dict(arrowstyle='->',
                                                    color=config['color'],
                                                    lw=arrow_width,
                                                    mutation_scale=20,
                                                    alpha=0.7 + normalized_q * 0.3))

                        # 在箭头末端添加 Q 值
                        offset_x = config['dx'] * 1.2
                        offset_y = config['dy'] * 1.2
                        ax.text(j + offset_x, i + offset_y,
                                f'{q_value:.2f}',
                                ha='center', va='center',
                                fontsize=7, fontweight='bold',
                                color=config['color'],
                                bbox=dict(boxstyle='round,pad=0.2',
                                          facecolor='white',
                                          edgecolor=config['color'],
                                          alpha=0.8))

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

        # Mark boom position with bomb shape
        bomb_body = plt.Circle((self.boom_state[1], self.boom_state[0]),
                               0.25, facecolor='none', edgecolor='black', linewidth=2,
                               label='Boom')
        ax.add_patch(bomb_body)

        ax.plot([self.boom_state[1] + 0.05, self.boom_state[1] + 0.25],
                [self.boom_state[0] - 0.05, self.boom_state[0] - 0.25],
                color='#8B4513', linewidth=3, solid_capstyle='round')

        spark = plt.Circle((self.boom_state[1] + 0.25, self.boom_state[0] - 0.25),
                           0.06, facecolor='#FF4500', edgecolor='none')
        ax.add_patch(spark)

        # 添加图例说明
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                       markersize=15, markeredgecolor='purple', label='Agent'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue',
                       markersize=12, markeredgecolor='black', label='Start'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green',
                       markersize=20, markeredgecolor='gold', label='Goal'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                       markersize=10, markeredgecolor='black', linewidth=2, label='Boom'),
            plt.Line2D([0], [0], color='blue', linewidth=2, label='上'),
            plt.Line2D([0], [0], color='green', linewidth=2, label='下'),
            plt.Line2D([0], [0], color='orange', linewidth=2, label='左'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='右'),
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  bbox_to_anchor=(1.3, 1), fontsize=10)

        # Add title
        ax.set_title('Grid World - Q 函数可视化 (箭头表示动作价值)',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()

# grid_world = GridWorld(4, 3)
# for cur_state in grid_world.getStates():
#     print(cur_state)
#
# # 随机生成V
# V = {state: np.random.uniform(-1, 1) for state in grid_world.getStates()}
# grid_world.render(V=V)
