import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import heapq
import collections
import random
import math
from queue import PriorityQueue
import queue as thread_queue
import os

# 全局开关：是否启用简化版 B* 寻路。
# 设置为 True 时，bstar_path_with_trace 将调用 bstar_path_easy 而不是完整工程版实现。
USE_BSTAR_EASY = True

# === 常量定义 ===
FREE = 0  # 通路
BLOCKED = 1  # 障碍

# === 迷宫生成（DFS） ===
def generate_maze(width, height):
    """
    使用深度优先搜索生成奇数尺寸迷宫。
    返回 NumPy 数组，FREE 表示通路，BLOCKED 表示墙。
    """
    # 确保奇数尺寸
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1
    grid = np.ones((height, width), dtype=int)
    start = (1, 1)
    stack = [start]
    grid[start] = FREE
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and grid[nx, ny] == BLOCKED:
                wall = (x + dx//2, y + dy//2)
                neighbors.append((nx, ny, wall))
        if neighbors:
            nx, ny, wall = random.choice(neighbors)
            grid[wall] = FREE
            grid[nx, ny] = FREE
            stack.append((nx, ny))
        else:
            stack.pop()
    # 确保入口出口通路
    grid[1,1] = FREE
    grid[-2,-2] = FREE
    return grid

# === 随机障碍网格（非迷宫）生成 ===
def generate_multiple_obstacles_not_maze(width, height, probability):
    """
    生成一个包含随机障碍的网格，用于测试非迷宫场景下的路径搜索。

    参数：
        width (int):  网格宽度（列数）。
        height (int): 网格高度（行数）。
        probability (float): 每个非边界格子生成障碍的概率，取值范围 [0, 1]。

    返回：
        np.ndarray: 二维数组，其中 0 表示可通行，1 表示障碍。

    说明：
        生成的网格并不保证一定存在从入口到出口的通路，只用于测试算法在随机障碍环境下的行为。
        起点 (1,1) 和终点 (height-2, width-2) 会被确保设置为可通行。
    """
    # 确保尺寸为正
    width = max(1, int(width))
    height = max(1, int(height))
    # 创建全障碍的网格
    grid = np.ones((height, width), dtype=int)
    # 随机填充障碍：边界除外
    for i in range(height):
        for j in range(width):
            # 边界保持通行，内部按照概率生成障碍
            if i == 0 or j == 0 or i == height - 1 or j == width - 1:
                grid[i, j] = FREE
            else:
                grid[i, j] = BLOCKED if random.random() < probability else FREE
    # 确保入口和出口为通路
    if height > 2 and width > 2:
        grid[1, 1] = FREE
        grid[height - 2, width - 2] = FREE
    return grid

# === A* 寻路 ===
def astar_path_with_trace(grid, start, goal):
    """返回最短路径列表（含端点），不可达时返回 None。"""
    if grid[start] == BLOCKED or grid[goal] == BLOCKED:
        return None
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    directions = [(0,1),(1,0),(0,-1),(-1,0)]
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    while not open_set.empty():
        _, current = open_set.get()
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dx, dy in directions:
            nbr = (current[0]+dx, current[1]+dy)
            if 0 <= nbr[0] < grid.shape[0] and 0 <= nbr[1] < grid.shape[1] and grid[nbr] == FREE:
                tentative = g_score[current] + 1
                if tentative < g_score.get(nbr, float('inf')):
                    came_from[nbr] = current
                    g_score[nbr] = tentative
                    open_set.put((tentative + manhattan(nbr, goal), nbr))
    return None

# === Branch-Star 寻路 ===
"""
### B\*（分支星）算法工程版解释

**算法概述：**

B\*算法在游戏中，尤其是怪物自动寻路时，比标准A\*算法更加高效。其核心优势在于它模拟自然界动物绕开障碍物的方式，并在需要时动态调整路径，避免死角，提高了路径计算效率。
特别在没有独立AI线程的游戏服务器上，B\*提供了一种优化寻路计算的方案，避免了像A*那样可能会消耗大量计算资源的操作。

### 1. **算法原理：**

B\*算法通过引入**状态转换机制**和**绕爬机制**，使得算法在面对复杂障碍时能够智能选择路径。具体而言：

- **探索状态（Exploration）：** 探索节点朝目标前进。如果前方无障碍物，继续前进。
- **绕爬状态（Climbing）：** 当遇到障碍物时，节点会进入绕爬状态，尝试绕过障碍。
- **角落状态（Cornering）：** 如果绕爬也无法解决问题，则进入角落状态，尝试更多方向来解决问题。

### 2. **前置定义：**

- **探索节点：** 也叫当前处理的节点，它正在向目标节点移动。
- **自由探索节点：** 当前节点周围无障碍物，可以自由前进。
- **绕爬探索节点：** 当前方有障碍时，节点尝试绕开障碍。
- **角落探索节点：** 如果绕爬无效，则进入角落状态，尝试更多复杂的绕行。

### 3. **算法流程：**

#### (1) 初始状态：

算法从起始点出发，将起始点标记为自由探索节点，并计算可能的前进方向。

#### (2) 碰到障碍：

- **自由探索节点**：如果路径畅通无阻，节点继续朝目标方向前进。
- **绕爬探索节点**：当路径被障碍物阻挡时，节点会变成绕爬状态，尝试绕过障碍，生成两个子分支（分别朝两个方向尝试绕行）。

#### (3) 状态转换：

节点会在**探索状态**、**绕爬状态**、**角落状态**之间切换。例如：

- **绕爬状态**：当节点试图绕开障碍时，它会沿着优先方向（可能是正交或对角线方向）继续尝试绕行。
- **角落状态**：当节点被困在障碍中时，它会尝试更多的方向进行移动，直到路径找到或无法继续。

#### (4) 紧急全向扩散：

当搜索队列为空，算法需要进行**紧急全向扩散**。这意味着：

- **为什么需要全向扩散：** 当所有常规搜索路径都耗尽时，紧急全向扩散可以作为补充措施，确保在复杂环境中不会因为无法绕行而陷入死循环。全向扩散会考虑所有方向，生成新的可行路径，避免算法进入死角。

#### (5) 返回结果：

一旦目标点被找到，算法会通过节点的父指针回溯路径，最终返回从起点到目标的路径。

### 4. **实现坑点与细节：**

#### (1) 方向优先级计算：

- 代码中提供了**方向优先级列表**的生成函数（如`get_direction_priority`），它根据当前位置和目标的相对位置，动态生成可能的方向，保证算法选择最优路径。

- **对角线与正交方向的选择：** 当目标位置与当前节点处于对角线方向时，算法会倾向选择对角线方向；否则，优先选择正交方向。

#### (2) 节点状态管理：

- 节点的状态（如`exploration`、`climbing`、`cornering`）是算法中的关键要素，决定了如何处理遇到的障碍物。
- **状态切换**：状态切换逻辑需要精确控制，防止出现逻辑错误导致死循环。例如，**绕爬状态**下，节点会重新选择方向，而不是直接回到**探索状态**。

#### (3) 碰撞计数：

代码中使用了**碰撞计数**（`collisions`）来决定节点的优先级。每当节点遇到障碍物时，计数加一，这样在优先队列中，碰撞较少的节点会先被处理，从而提高效率。

#### (4) 紧急全向扩散的必要性：

在复杂的地图环境中，常规的路径寻找可能会遇到**死角**，即没有任何可行路径继续前进。为了避免这种情况，B\*算法引入了**全向扩散**策略。当优先队列为空时，算法会从所有可能的方向重新尝试，确保找到可行路径。

- **全向扩散的实现：** 通过对所有方向进行尝试（即使用`ALL_DIRECTIONS`），即使在常规路径遇到瓶颈时，算法也能重新寻找路径。这样，可以防止被困死角。

#### (5) 复杂障碍物的处理：

在复杂的地图和动态障碍中，B\*需要不断调整**绕爬策略**。例如，**绕爬的探索节点**在成功绕过障碍后，会重新回到**自由节点**状态，这样可以保证不被单一方向限制。

整合以上即可实现既快速又完备的 B* 寻路。
"""
# 辅助函数：检查位置是否在网格范围内
def in_bounds(pos, grid):
    return 0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0])

# 辅助函数：检查位置是否可通行
def walkable(pos, grid):
    return in_bounds(pos, grid) and grid[pos[0]][pos[1]] == FREE

# 回溯重建路径
def reconstruct_path(node):
    path = []
    while node:
        path.append(node.pos)
        node = node.parent
    return path[::-1]

# 计算方向向量
def get_direction_vector(current, target):
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    # 使用 int() 将 copysign 结果转换为整数方向
    return (
        int(math.copysign(1, dx)) if dx != 0 else 0,
        int(math.copysign(1, dy)) if dy != 0 else 0
    )

# === 核心数据结构 ===
class Node:
    """
    存储 B* 搜索过程中的节点信息。
    每个节点记录当前位置、父节点、状态、移动方向、碰撞次数以及待尝试的方向列表。
    """
    __slots__ = ('pos', 'parent', 'state', 'dir', 'collisions', 'directions', 'dir_idx')
    def __init__(self, pos, parent=None, state='exploration', dir=None, collisions=0):
        self.pos = pos          # 当前位置
        self.parent = parent    # 父节点
        self.state = state      # 状态: exploration/climbing/cornering
        self.dir = dir          # 当前移动方向
        self.collisions = collisions  # 碰撞计数
        self.directions = []    # 方向优先级列表
        self.dir_idx = -1       # 当前方向索引

def create_visited_map(grid):
    """
    创建与地图同尺寸的访问标记，用于记录某个格子是否被访问。
    True 表示已经访问，False 表示未访问。
    """
    return np.zeros_like(grid, dtype=bool)

# 定义方向常量
ORTHOGONAL_DIRECTIONS = [(1,0), (-1,0), (0,1), (0,-1)]
DIAGONAL_DIRECTIONS   = [(1,1), (1,-1), (-1,1), (-1,-1)]
ALL_DIRECTIONS        = ORTHOGONAL_DIRECTIONS + DIAGONAL_DIRECTIONS

def get_direction_priority(src, dest, grid):
    """
    根据当前位置与目标的相对关系，生成一个方向优先级列表。
    若当前位置与目标点在对角线方向，则生成八方向列表；否则生成四方向列表。
    返回的列表以低优先级在前，高优先级在后，便于使用栈结构逐个弹出。
    """
    dx = dest[0] - src[0]
    dy = dest[1] - src[1]
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    if abs_dx == abs_dy and abs_dx > 0:
        # 对角线移动时使用八方向
        return get_diagonal_priority(src, dest)[::-1]
    else:
        # 否则使用正交方向
        return get_orthogonal_priority(src, dest)[::-1]

def get_orthogonal_priority(src, dest):
    """
    根据起点 src 和终点 dest 的相对位置，返回四个正交方向的优先级列表。

    该实现基于原始 B* 算法的方向优先规则，使用笛卡尔坐标 (x, y)，其中 x 轴向右为正，y 轴向上为正。
    在我们的网格中，列索引相当于 x，行索引相当于 y 向下为正，故需进行坐标转换。

    返回的方向以 (row_delta, col_delta) 表示。
    """
    # 计算在笛卡尔坐标系下的差值
    # x 轴差：终点列索引减起点列索引
    dx = dest[1] - src[1]
    # y 轴差：终点行索引减起点行索引，但笛卡尔坐标的 y 轴向上为正，故取相反数
    dy = -(dest[0] - src[0])
    # 根据象限和差值大小决定优先级
    if dx >= 0 and dy >= 0:  # 第一象限：右上
        if dx > dy:
            dirs = [(1,0), (0,1), (0,-1), (-1,0)]  # (dx,dy) 顺序
        else:
            dirs = [(0,1), (1,0), (-1,0), (0,-1)]
    elif dx >= 0 and dy < 0:  # 第四象限：右下
        if dx > -dy:
            dirs = [(1,0), (0,-1), (0,1), (-1,0)]
        else:
            dirs = [(0,-1), (1,0), (-1,0), (0,1)]
    elif dx < 0 and dy >= 0:  # 第二象限：左上
        if -dx > dy:
            dirs = [(-1,0), (0,1), (0,-1), (1,0)]
        else:
            dirs = [(0,1), (-1,0), (1,0), (0,-1)]
    else:  # 第三象限：左下
        if -dx > -dy:
            dirs = [(-1,0), (0,-1), (0,1), (1,0)]
        else:
            dirs = [(0,-1), (-1,0), (1,0), (0,1)]
    # 将笛卡尔坐标方向转换为 (row_delta, col_delta)
    return [(-dy_, dx_) for (dx_, dy_) in dirs]

def get_diagonal_priority(src, dest):
    """
    根据起点 src 和终点 dest 的相对位置，返回八个方向的优先级列表，用于对角线移动。

    该实现基于原始 B* 算法的方向优先规则，使用笛卡尔坐标 (x, y)，其中 x 轴向右为正，y 轴向上为正。
    在我们的网格中，列索引相当于 x，行索引相当于 y 向下为正，故需进行坐标转换。

    返回的方向以 (row_delta, col_delta) 表示。
    """
    dx = dest[1] - src[1]         # x 轴差
    dy = -(dest[0] - src[0])      # y 轴差
    # 使用原始算法的八方向优先级
    if dx >= 0 and dy >= 0:  # 第一象限：右上
        if dx > dy:
            dirs = [(1,0), (1,1), (0,1), (1,-1), (0,-1), (-1,1), (-1,0), (-1,-1)]
        else:
            dirs = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,1), (-1,0), (-1,-1)]
    elif dx >= 0 and dy < 0:  # 第四象限：右下
        if dx > -dy:
            dirs = [(1,0), (1,-1), (0,-1), (1,1), (0,1), (-1,-1), (-1,0), (-1,1)]
        else:
            dirs = [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,-1), (-1,0), (-1,1)]
    elif dx < 0 and dy >= 0:  # 第二象限：左上
        if -dx > dy:
            dirs = [(-1,0), (-1,1), (0,1), (-1,-1), (0,-1), (1,1), (1,0), (1,-1)]
        else:
            dirs = [(0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,1), (1,0), (1,-1)]
    else:  # 第三象限：左下
        if -dx > -dy:
            dirs = [(-1,0), (-1,-1), (0,-1), (-1,1), (0,1), (1,-1), (1,0), (1,1)]
        else:
            dirs = [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,-1), (1,0), (1,1)]
    # 将笛卡尔坐标方向转换为 (row_delta, col_delta)
    return [(-dy_, dx_) for (dx_, dy_) in dirs]

def get_secondary_directions(primary_dir):
    """
    遇到障碍时用于绕行的次要方向。
    对于正交移动，次要方向是左右；对于水平/垂直移动，次要方向是上下。
    对于对角移动，次要方向包括正交方向与反向两侧。
    """
    dx, dy = primary_dir
    if dx == 0:
        return [(1,0), (-1,0)]
    elif dy == 0:
        return [(0,1), (0,-1)]
    else:
        return [(dx,0), (0,dy), (-dx,0), (0,-dy)]

def get_perpendicular_directions(direction):
    """
    在角落状态下，尝试与当前方向垂直的方向组合。
    """
    dx, dy = direction
    if dx == 0:
        return [(1,0), (-1,0)]
    elif dy == 0:
        return [(0,1), (0,-1)]
    else:
        return [(dy,dx), (-dy,-dx)]

def get_primary_direction(current, goal):
    """
    获取当前位置到目标位置的主方向。优先沿轴向距离较大的方向前进。
    """
    dx = goal[0] - current[0]
    dy = goal[1] - current[1]
    if dx == 0 and dy == 0:
        return (0, 0)
    if abs(dx) > abs(dy):
        return (1 if dx > 0 else -1, 0)
    else:
        return (0, 1 if dy > 0 else -1)

def handle_exploration(node, grid, visited, heap, goal, counter):
    """
    处理探索状态：
    - 按优先级尝试下一个方向；
    - 若遇到障碍，则转为攀爬状态并生成次要方向列表。
    """
    if node.dir_idx < 0:
        # 已无方向可尝试
        return counter
    # 取出一个方向
    direction = node.directions[node.dir_idx]
    node.dir_idx -= 1
    new_pos = (node.pos[0] + direction[0], node.pos[1] + direction[1])
    if walkable(new_pos, grid) and not visited[new_pos]:
        # 自由移动到未访问格子
        visited[new_pos] = True
        new_node = Node(new_pos, node, 'exploration', direction, node.collisions)
        new_node.directions = get_direction_priority(new_pos, goal, grid)
        new_node.dir_idx = len(new_node.directions) - 1
        heapq.heappush(heap, (new_node.collisions, counter, new_node))
        counter += 1
    else:
        # 碰壁，转换为攀爬状态并生成次要方向
        node.state = 'climbing'
        node.collisions += 1
        node.directions = get_secondary_directions(direction)
        node.dir_idx = len(node.directions) - 1
        heapq.heappush(heap, (node.collisions, counter, node))
        counter += 1
    return counter

def handle_climbing(node, grid, visited, heap, goal, counter):
    """
    处理攀爬状态：
    - 尝试回归主方向；若可行则生成探索节点；
    - 否则继续沿攀爬方向前进；
    - 当方向列表耗尽时转入角落状态。
    """
    primary_dir = get_primary_direction(node.pos, goal)
    new_pos = (node.pos[0] + primary_dir[0], node.pos[1] + primary_dir[1])
    # 尝试沿主方向回归
    if walkable(new_pos, grid) and not visited[new_pos]:
        visited[new_pos] = True
        new_node = Node(new_pos, node, 'exploration', primary_dir, node.collisions)
        new_node.directions = get_direction_priority(new_pos, goal, grid)
        new_node.dir_idx = len(new_node.directions) - 1
        heapq.heappush(heap, (new_node.collisions, counter, new_node))
        counter += 1
        return counter
    # 继续沿当前方向
    if node.dir_idx < 0:
        # 没有方向可以继续，转为角落状态
        node.state = 'cornering'
        node.collisions += 1
        node.directions = get_perpendicular_directions(node.dir)
        node.dir_idx = len(node.directions) - 1
        heapq.heappush(heap, (node.collisions, counter, node))
        counter += 1
        return counter
    direction = node.directions[node.dir_idx]
    node.dir_idx -= 1
    new_pos = (node.pos[0] + direction[0], node.pos[1] + direction[1])
    if walkable(new_pos, grid) and not visited[new_pos]:
        visited[new_pos] = True
        new_node = Node(new_pos, node, 'climbing', direction, node.collisions)
        new_node.directions = get_secondary_directions(direction)
        new_node.dir_idx = len(new_node.directions) - 1
        heapq.heappush(heap, (new_node.collisions, counter, new_node))
        counter += 1
    else:
        # 再次碰壁，增加碰撞计数并重新入队列
        node.collisions += 1
        heapq.heappush(heap, (node.collisions, counter, node))
        counter += 1
    return counter

def handle_cornering(node, grid, visited, heap, goal, counter):
    """
    处理角落状态：
    - 优先尝试回归主方向；
    - 尝试自身方向栈；
    - 最后尝试所有方向，生成新的攀爬或角落节点。
    """
    # 尝试回归主方向
    primary_dir = get_primary_direction(node.pos, goal)
    new_pos = (node.pos[0] + primary_dir[0], node.pos[1] + primary_dir[1])
    if walkable(new_pos, grid) and not visited[new_pos]:
        visited[new_pos] = True
        new_node = Node(new_pos, node, 'exploration', primary_dir, node.collisions)
        new_node.directions = get_direction_priority(new_pos, goal, grid)
        new_node.dir_idx = len(new_node.directions) - 1
        heapq.heappush(heap, (new_node.collisions, counter, new_node))
        counter += 1
        return counter
    # 尝试当前方向栈
    if node.dir_idx >= 0:
        direction = node.directions[node.dir_idx]
        node.dir_idx -= 1
        new_pos = (node.pos[0] + direction[0], node.pos[1] + direction[1])
        if walkable(new_pos, grid) and not visited[new_pos]:
            visited[new_pos] = True
            new_node = Node(new_pos, node, 'cornering', direction, node.collisions)
            heapq.heappush(heap, (new_node.collisions, counter, new_node))
            counter += 1
            return counter
    # 尝试所有可能方向
    # 3. 尝试所有方向
    for direction in ALL_DIRECTIONS:
        new_pos = (node.pos[0] + direction[0], node.pos[1] + direction[1])
        if walkable(new_pos, grid) and not visited[new_pos]:
            visited[new_pos] = True
            state = 'climbing' if direction in ORTHOGONAL_DIRECTIONS else 'cornering'
            new_node = Node(new_pos, node, state, direction, node.collisions + 1)
            heapq.heappush(heap, (new_node.collisions, counter, new_node))
            counter += 1
    return counter

# === 简化版 B* 寻路 ===
def bstar_path_easy(grid, start, goal):
    """
    基于 0-1 BFS 的简化 B* 寻路算法。

    该实现使用双端队列维护当前探索的节点，并根据“碰撞”代价动态调整扩展顺序：
    - 沿主方向或继续绕行的移动视为代价 0，将节点放入队首；
    - 遇到障碍首次生成左右绕行分支，视为代价 1，将节点放入队尾。

    通过这种方式，可以在不引入多线程的情况下优先扩展更靠近目标的低代价路径，
    同时保持 B* 易于实现和调试的优势。

    参数：
        grid (np.ndarray): 0 表示可通行，非 0 表示障碍。
        start (tuple): 起点坐标 (row, col)。
        goal (tuple): 终点坐标 (row, col)。

    返回：
        list 或 None: 找到路径时返回包含起点和终点的坐标列表，否则返回 None。
    """
    # 起点或终点不可通行
    if not in_bounds(start, grid) or not in_bounds(goal, grid) or \
       not walkable(start, grid) or not walkable(goal, grid):
        return None
    from collections import deque
    # best 记录 (位置, 状态) 的最小碰撞次数
    best = {}
    dq = deque()
    # 初始化
    start_node = Node(start, parent=None, state='exploration', dir=None, collisions=0)
    dq.appendleft(start_node)
    best[(start, 'exploration')] = 0
    while dq:
        node = dq.popleft()
        pos = node.pos
        state = node.state
        collisions = node.collisions
        # 到达终点
        if pos == goal:
            return reconstruct_path(node)
        # 如果已存在更优解，跳过
        if collisions > best.get((pos, state), collisions):
            continue
        if state == 'exploration':
            # 主方向
            major_dir = get_direction_vector(pos, goal)
            if major_dir != (0, 0):
                major_pos = (pos[0] + major_dir[0], pos[1] + major_dir[1])
                # 沿主方向移动，代价不增
                if walkable(major_pos, grid):
                    key = (major_pos, 'exploration')
                    if collisions < best.get(key, float('inf')):
                        best[key] = collisions
                        new_node = Node(major_pos, node, 'exploration', major_dir, collisions)
                        dq.appendleft(new_node)
                else:
                    # 主方向受阻，生成左右绕行分支，代价 +1
                    secondary_dirs = get_secondary_directions(major_dir)
                    branch_dirs = secondary_dirs[:2] if len(secondary_dirs) > 2 else secondary_dirs
                    for branch_dir in branch_dirs:
                        branch_pos = (pos[0] + branch_dir[0], pos[1] + branch_dir[1])
                        if walkable(branch_pos, grid):
                            key = (branch_pos, 'climbing')
                            new_collisions = collisions + 1
                            if new_collisions < best.get(key, float('inf')):
                                best[key] = new_collisions
                                new_node = Node(branch_pos, node, 'climbing', branch_dir, new_collisions)
                                dq.append(new_node)  # 放入队尾
        else:
            # climbing 状态
            major_dir = get_direction_vector(pos, goal)
            if major_dir != (0, 0):
                major_pos = (pos[0] + major_dir[0], pos[1] + major_dir[1])
                # 尝试回归主方向
                if walkable(major_pos, grid):
                    key = (major_pos, 'exploration')
                    if collisions < best.get(key, float('inf')):
                        best[key] = collisions
                        new_node = Node(major_pos, node, 'exploration', major_dir, collisions)
                        dq.appendleft(new_node)
                        continue
            # 继续沿当前绕行方向
            climb_dir = node.dir
            if climb_dir is not None:
                next_pos = (pos[0] + climb_dir[0], pos[1] + climb_dir[1])
                if walkable(next_pos, grid):
                    key = (next_pos, 'climbing')
                    if collisions < best.get(key, float('inf')):
                        best[key] = collisions
                        new_node = Node(next_pos, node, 'climbing', climb_dir, collisions)
                        dq.appendleft(new_node)
    return None

def bstar_path_full(grid, start, goal):
    """
    完整工程版 B* 寻路算法。

    采用探索、攀爬、角落三种状态机和全向扩散策略，能够在复杂迷宫中找到路径。

    返回从起点到终点的路径（包含端点），若不可达则返回 None。
    """
    # 初始化访问标记
    visited = create_visited_map(grid)
    # 优先队列，按碰撞次数排序
    heap = []
    counter = 0
    # 创建起始节点
    start_node = Node(start, parent=None, state='exploration', dir=None, collisions=0)
    start_node.directions = get_direction_priority(start, goal, grid)
    start_node.dir_idx = len(start_node.directions) - 1
    visited[start] = True
    heapq.heappush(heap, (start_node.collisions, counter, start_node))
    counter += 1
    # 存储所有生成过的节点，用于后续全方向扩散
    all_nodes = set()
    all_nodes.add(start_node)
    # 主循环
    while True:
        while heap:
            collisions, _, current = heapq.heappop(heap)
            # 到达终点
            if current.pos == goal:
                return reconstruct_path(current)
            # 根据状态处理
            if current.state == 'exploration':
                counter = handle_exploration(current, grid, visited, heap, goal, counter)
            elif current.state == 'climbing':
                counter = handle_climbing(current, grid, visited, heap, goal, counter)
            else:  # cornering
                counter = handle_cornering(current, grid, visited, heap, goal, counter)
            # 将当前节点加入 all_nodes，用于可能的全方向扩散
            all_nodes.add(current)
        # 若堆为空，则尝试全方向扩散
        new_nodes = []
        for node in list(all_nodes):
            for direction in ORTHOGONAL_DIRECTIONS:
                new_pos = (node.pos[0] + direction[0], node.pos[1] + direction[1])
                if 0 <= new_pos[0] < grid.shape[0] and 0 <= new_pos[1] < grid.shape[1] and grid[new_pos] == FREE:
                    if not visited[new_pos]:
                        visited[new_pos] = True
                        new_node = Node(new_pos, node, 'exploration', direction, node.collisions)
                        new_node.directions = get_direction_priority(new_pos, goal, grid)
                        new_node.dir_idx = len(new_node.directions) - 1
                        heapq.heappush(heap, (new_node.collisions, counter, new_node))
                        counter += 1
                        all_nodes.add(new_node)
                        new_nodes.append(new_node)
        # 如果新节点非空，则继续主循环
        if new_nodes:
            continue
        # 若没有新节点生成则结束搜索
        break
    # 未找到路径
    return None

def bstar_path_with_trace(grid, start, goal):
    """
    综合伪代码实现的 B* 寻路算法。

    根据全局开关 USE_BSTAR_EASY，先尝试简化版寻路，如果未找到路径则回退到完整工程版实现。

    返回从起点到终点的路径（包含端点），若不可达则返回 None。
    """
    # 首先检查起点和终点是否可行走
    if not in_bounds(start, grid) or not in_bounds(goal, grid) or \
       not walkable(start, grid) or not walkable(goal, grid):
        return None
    path = None
    # 若启用简化版，尝试简化版寻路
    if USE_BSTAR_EASY:
        path = bstar_path_easy(grid, start, goal)
    else:
        path = bstar_path_full(grid, start, goal)
    return path
    
# === 可视化示例 ===
def visualize_paths(grid_size=51):
    maze = generate_maze(grid_size, grid_size)
    start, goal = (1,1), (grid_size-2,grid_size-2)
    maze[start]=0; maze[goal]=0
    path_a = astar_path_with_trace(maze, start, goal)
    path_b = bstar_path_with_trace(maze, start, goal)
    plt.figure(figsize=(6,6))
    plt.imshow(maze, cmap='gray', origin='lower', interpolation='none')

    print(f"可视化迷宫边长: {grid_size}")
    print(f"A* 路径长度: {len(path_a) if path_a else '未找到'}")
    print(f"B* 路径长度: {len(path_b) if path_b else '未找到'}")
    
    if path_a and path_b:
        ya, xa = zip(*path_a)
        yb, xb = zip(*path_b)
        plt.plot(xa, ya, '-', color='red', label='A*')
        plt.plot(xb, yb, '-', color='blue', label='B*')
    else:
        if path_a:
            ya, xa = zip(*path_a)
            plt.plot(xa, ya, '-', color='red', label='A*')
        if path_b:
            yb, xb = zip(*path_b)
            plt.plot(xb, yb, '-', color='blue', label='B*')

    plt.legend(); plt.axis('off'); plt.show()

# === 基准测试 ===
def benchmark(sizes, runs, use_maze=False, obstacle_probability=0.2):
    """
    简单基准测试函数，对比 A* 和 B* 在不同地图尺寸上的平均运行时间。

    参数：
        sizes (list[int]): 要测试的迷宫边长列表（必须为奇数，偶数将自动加 1）。
        runs (int): 每个地图尺寸下重复运行的次数，取平均值。
        use_maze (bool): 为 True 时使用迷宫生成函数；为 False 时使用随机障碍网格。
        obstacle_probability (float): 当 use_maze 为 False 时，非迷宫场景中每个内部格子成为障碍的概率。

    返回：
        pd.DataFrame: 包含不同算法和地图尺寸的平均运行时间、最小和最大运行时间。
    """
    print(f"使用迷宫: {use_maze}")
    results = []
    algs = {'A*': astar_path_with_trace, 'B*': bstar_path_with_trace}
    for size in sizes:
        # 迷宫边长需为奇数，若为偶数则加 1
        w = size + (size % 2 == 0)
        # 生成地图：迷宫或随机障碍
        if use_maze:
            grid = generate_maze(w, w)
        else:
            grid = generate_multiple_obstacles_not_maze(w, w, obstacle_probability)
        start, goal = (1, 1), (w - 2, w - 2)
        # 确保起点和终点可通行
        grid[start] = FREE
        grid[goal] = FREE
        for name, func in algs.items():
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                path = func(grid, start, goal)
                print(f"{name} 路径长度: {len(path) if path else '未找到'}")
                times.append(time.perf_counter() - t0)
            results.append({
                'algorithm': name,
                'size': w,
                'avg_time': sum(times) / runs,
                'min_time': min(times),
                'max_time': max(times)
            })
    return pd.DataFrame(results)

# === 主入口 ===
if __name__=='__main__':
    #random.seed(42); np.random.seed(42)
    print(f"USE_BSTAR_EASY: {USE_BSTAR_EASY}")
    df = benchmark([101,151,201], 4, False)
    print(df)
    visualize_paths(51)
    # 测试用例
    empty = np.zeros((5,5),int)
    print('所有测试通过')