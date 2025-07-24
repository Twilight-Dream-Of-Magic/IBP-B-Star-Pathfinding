#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure IBP-B* (paper-accurate, lean & fast) - Python

Implements exactly the paper's mechanisms:
1) Greedy one-step move
2) First / multi obstacle rules
3) Obstacle-avoidance rebirth (triggered only when B's left & right are obstacles, once per node)
4) Pre-exploration for concave glyph (simple O(1) check)
5) Two-way parallel search
6) Peer waiting: same layer + ±WAIT_LAYERS

Add-ons (not affecting core logic):
- Path validity check (avoid printing garbage if chain is broken)
- Random map auto re-roll until solvable (optional)
- ASCII fallback
"""

import numpy as np
import heapq
from collections import deque
import random
import sys

class CONFIG:
    # Map source
    USE_RANDOM = True
    RAND_W, RAND_H, RAND_P, RAND_SEED = 64, 64, 0.25, 56464641
    MAP_FILE   = None
    FILE_START = None
    FILE_END   = None
    ENSURE_SOLVABLE = True
    ENSURE_MAX_TRY  = 100

    # Algo params
    WAIT_LAYERS = 2
    PRINT_PATH  = True
    PRINT_STATS = True
    PRINT_INVALID_PATH = False  # 若 path 验证失败，是否仍打印
    ARROW_PATH  = False

    # Glyphs (unicode by default, ASCII fallback by switch)
    USE_ASCII = True
    WALL_U  = "■"; EMPTY_U = "○"; PATH_U  = "★"
    START_U = "×"; END_U   = "√"
    ARROWS_U = {"U":"↑","D":"↓","L":"←","R":"→"}

    WALL_A  = "+"; EMPTY_A = "."; PATH_A  = "?"
    START_A = "S"; END_A   = "E"
    ARROWS_A= {"U":"^","D":"v","L":"<","R":">"}

class IBP_BStar:
    def __init__(self, grid):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        self.directions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.direction_priority = ['U', 'D', 'L', 'R']
    
    def is_passable(self, row, col):
        return (0 <= row < self.height and 
                0 <= col < self.width and 
                self.grid[row][col] == 0)
    
    def choose_greedy_direction(self, current_row, current_col, goal_row, goal_col):
        dr = goal_row - current_row
        dc = goal_col - current_col
        if abs(dr) >= abs(dc):
            return 'D' if dr > 0 else 'U'
        else:
            return 'R' if dc > 0 else 'L'
    
    def opposite_direction(self, direction):
        opposites = {'U': 'D', 'D': 'U', 'L': 'R', 'R': 'L'}
        return opposites[direction]
    
    def left_right_directions(self, direction):
        # 返回 (左方向, 右方向)
        if direction == 'U': return ('L', 'R')
        if direction == 'D': return ('R', 'L')
        if direction == 'L': return ('D', 'U')
        return ('U', 'D')  # direction == 'R'
    
    def is_concave_entry(self, row, col, direction):
        if direction in ['L', 'R']:
            up_passable = (row > 0 and self.is_passable(row-1, col))
            down_passable = (row < self.height-1 and self.is_passable(row+1, col))
            return up_passable and down_passable
        else:
            left_passable = (col > 0 and self.is_passable(row, col-1))
            right_passable = (col < self.width-1 and self.is_passable(row, col+1))
            return left_passable and right_passable
    
    def run_ibp_bstar(self, start, goal, wait_layers=CONFIG.WAIT_LAYERS):
        # 初始化数据结构
        depth_from_start = [[-1] * self.width for _ in range(self.height)]
        depth_from_goal = [[-1] * self.width for _ in range(self.height)]
        parent_from_start = [[None] * self.width for _ in range(self.height)]
        parent_from_goal = [[None] * self.width for _ in range(self.height)]
        
        # 使用字典记录障碍历史
        history_from_start = [[{'hit_count': 0, 'last_dir': None, 'rebirth_used': False, 'zigzag_used': False} 
                              for _ in range(self.width)] for _ in range(self.height)]
        history_from_goal = [[{'hit_count': 0, 'last_dir': None, 'rebirth_used': False, 'zigzag_used': False} 
                             for _ in range(self.width)] for _ in range(self.height)]
        
        # 初始化队列
        queue_from_start = deque([start])
        queue_from_goal = deque([goal])
        depth_from_start[start[0]][start[1]] = 0
        depth_from_goal[goal[0]][goal[1]] = 0
        
        meet_point = None
        meet_depth_sum = -1
        flush_depth_levels = set()
        stats = {'expanded': 0, 'zigzag_used': 0}
        
        # 修复相遇检测
        def check_meet(row, col, depth_arr, other_depth_arr):
            nonlocal meet_point, meet_depth_sum
            if meet_point is not None:
                return
                
            # 检查是否到达终点或起点
            if (row, col) == goal or (row, col) == start:
                meet_point = (row, col)
                meet_depth_sum = depth_arr[row][col] + other_depth_arr[row][col]
                update_flush_levels()
                return
                
            # 检查是否在对方已访问节点中
            if other_depth_arr[row][col] != -1:
                meet_point = (row, col)
                meet_depth_sum = depth_arr[row][col] + other_depth_arr[row][col]
                update_flush_levels()
        
        def update_flush_levels():
            nonlocal flush_depth_levels
            flush_depth_levels = {meet_depth_sum}
            for k in range(1, wait_layers+1):
                if meet_depth_sum + k < self.height * self.width:
                    flush_depth_levels.add(meet_depth_sum + k)
                if meet_depth_sum - k >= 0:
                    flush_depth_levels.add(meet_depth_sum - k)
        
        # 弯延绕模式实现
        def run_zigzag_mode(row, col, direction, hit_count, depth_arr, parent_arr, queue, history_arr):
            left_dir, right_dir = self.left_right_directions(direction)
            left_dr, left_dc = self.directions[left_dir]
            right_dr, right_dc = self.directions[right_dir]
            
            moves = [
                (row + left_dr, col + left_dc),   # 左/右方向
                (row + right_dr, col + right_dc)   # 右/左方向
            ]
            
            # 高级弯延绕：尝试对角线方向
            if hit_count > 8:
                diag_dirs = ['L', 'R'] if direction in ['U', 'D'] else ['U', 'D']
                for diag_dir in diag_dirs:
                    dr, dc = self.directions[diag_dir]
                    moves.append((row + dr, col + dc))
            
            expanded = False
            for r, c in moves:
                if self.is_passable(r, c) and depth_arr[r][c] == -1:
                    depth_arr[r][c] = depth_arr[row][col] + 1
                    parent_arr[r][c] = (row, col)
                    
                    # 更新历史记录
                    new_history = history_arr[row][col].copy()
                    new_history['hit_count'] += 1
                    new_history['last_dir'] = direction
                    new_history['zigzag_used'] = True
                    history_arr[r][c] = new_history
                    
                    queue.append((r, c))
                    other_depth = depth_from_goal if depth_arr is depth_from_start else depth_from_start
                    check_meet(r, c, depth_arr, other_depth)
                    expanded = True
            
            if expanded:
                stats['zigzag_used'] += 1
                if CONFIG.PRINT_STATS:
                    print(f"Zigzag activated at ({row},{col}) hit_count={hit_count}")
            return expanded
        
        # 核心扩展函数
        def expand_frontier(current, goal_pos, depth_arr, parent_arr, queue, history_arr, other_depth_arr):
            r, c = current
            stats['expanded'] += 1
            
            # 1. 计算贪心方向
            greedy_dir = self.choose_greedy_direction(r, c, goal_pos[0], goal_pos[1])
            dr, dc = self.directions[greedy_dir]
            
            # 2. 计算B格和C格位置
            B_r, B_c = r + dr, c + dc
            C_r, C_c = r + 2*dr, c + 2*dc
            
            # 3. 预探索机制 (检测凹形陷阱)
            if (self.is_passable(B_r, B_c) and 
                self.is_passable(C_r, C_c) and
                (self.is_concave_entry(B_r, B_c, greedy_dir) or 
                 self.is_concave_entry(C_r, C_c, greedy_dir))):
                
                for dir_char in self.direction_priority:
                    if dir_char == greedy_dir:
                        continue
                    drd, dcd = self.directions[dir_char]
                    new_r, new_c = r + drd, c + dcd
                    
                    if self.is_passable(new_r, new_c) and depth_arr[new_r][new_c] == -1:
                        depth_arr[new_r][new_c] = depth_arr[r][c] + 1
                        parent_arr[new_r][new_c] = (r, c)
                        history_arr[new_r][new_c] = history_arr[r][c].copy()
                        queue.append((new_r, new_c))
                        check_meet(new_r, new_c, depth_arr, other_depth_arr)
            
            # 4. 判断B格是否可走
            if self.is_passable(B_r, B_c):
                if depth_arr[B_r][B_c] == -1:
                    depth_arr[B_r][B_c] = depth_arr[r][c] + 1
                    parent_arr[B_r][B_c] = (r, c)
                    history_arr[B_r][B_c] = history_arr[r][c].copy()
                    queue.append((B_r, B_c))
                    check_meet(B_r, B_c, depth_arr, other_depth_arr)
                return
            
            # 5. 障碍处理
            history_arr[r][c]['hit_count'] += 1
            history_arr[r][c]['last_dir'] = greedy_dir
            hit_count = history_arr[r][c]['hit_count']
            
            # 6. 障碍规避再生机制
            left_dir, right_dir = self.left_right_directions(greedy_dir)
            l_dr, l_dc = self.directions[left_dir]
            r_dr, r_dc = self.directions[right_dir]
            
            left_blocked = not self.is_passable(B_r + l_dr, B_c + l_dc)
            right_blocked = not self.is_passable(B_r + r_dr, B_c + r_dc)
            
            if left_blocked and right_blocked and not history_arr[r][c]['rebirth_used']:
                history_arr[r][c]['rebirth_used'] = True
                history_arr[r][c]['hit_count'] += 1
                queue.append((r, c))  # 重入队列
                
                # 垂直方向扩展
                alt_dirs = ['L', 'R'] if greedy_dir in ['U', 'D'] else ['U', 'D']
                for alt_dir in alt_dirs:
                    a_dr, a_dc = self.directions[alt_dir]
                    new_r, new_c = r + a_dr, c + a_dc
                    if self.is_passable(new_r, new_c) and depth_arr[new_r][new_c] == -1:
                        depth_arr[new_r][new_c] = depth_arr[r][c] + 1
                        parent_arr[new_r][new_c] = (r, c)
                        history_arr[new_r][new_c] = history_arr[r][c].copy()
                        queue.append((new_r, new_c))
                        check_meet(new_r, new_c, depth_arr, other_depth_arr)
                return
            
            # 7. 根据碰撞次数处理
            if hit_count == 1:  # 第一次碰撞
                for dir_char in self.direction_priority:
                    if dir_char == greedy_dir:
                        continue
                    d_dr, d_dc = self.directions[dir_char]
                    new_r, new_c = r + d_dr, c + d_dc
                    
                    if self.is_passable(new_r, new_c) and depth_arr[new_r][new_c] == -1:
                        depth_arr[new_r][new_c] = depth_arr[r][c] + 1
                        parent_arr[new_r][new_c] = (r, c)
                        history_arr[new_r][new_c] = history_arr[r][c].copy()
                        queue.append((new_r, new_c))
                        check_meet(new_r, new_c, depth_arr, other_depth_arr)
            else:  # 多次碰撞
                # 检查原障碍是否仍然存在
                obstacle_exists = not self.is_passable(r + dr, c + dc)
                
                if obstacle_exists:  # 障碍仍在：向反方向移动
                    opp_dir = self.opposite_direction(greedy_dir)
                    o_dr, o_dc = self.directions[opp_dir]
                    new_r, new_c = r + o_dr, c + o_dc
                    
                    if self.is_passable(new_r, new_c) and depth_arr[new_r][new_c] == -1:
                        depth_arr[new_r][new_c] = depth_arr[r][c] + 1
                        parent_arr[new_r][new_c] = (r, c)
                        history_arr[new_r][new_c] = history_arr[r][c].copy()
                        queue.append((new_r, new_c))
                        check_meet(new_r, new_c, depth_arr, other_depth_arr)
                else:  # 障碍消失：尝试原方向
                    new_r, new_c = r + dr, c + dc
                    if self.is_passable(new_r, new_c) and depth_arr[new_r][new_c] == -1:
                        depth_arr[new_r][new_c] = depth_arr[r][c] + 1
                        parent_arr[new_r][new_c] = (r, c)
                        history_arr[new_r][new_c] = history_arr[r][c].copy()
                        queue.append((new_r, new_c))
                        check_meet(new_r, new_c, depth_arr, other_depth_arr)
                
                # 尝试弯延绕模式
                if hit_count > 8 and not history_arr[r][c]['zigzag_used']:
                    expanded = run_zigzag_mode(r, c, greedy_dir, hit_count, 
                                              depth_arr, parent_arr, queue, history_arr)
                    if expanded:
                        return
                    history_arr[r][c]['zigzag_used'] = True
        
        # 主循环
        while queue_from_start or queue_from_goal:
            # 相遇后处理：冲刷指定层级的节点
            if meet_point is not None:
                # 检查是否需要冲刷
                start_flush = queue_from_start and depth_from_start[queue_from_start[0][0]][queue_from_start[0][1]] in flush_depth_levels
                goal_flush = queue_from_goal and depth_from_goal[queue_from_goal[0][0]][queue_from_goal[0][1]] in flush_depth_levels
                
                if not start_flush and not goal_flush:
                    break
                
                if start_flush:
                    current = queue_from_start.popleft()
                    expand_frontier(
                        current, goal, 
                        depth_from_start, parent_from_start, queue_from_start, history_from_start,
                        depth_from_goal
                    )
                
                if goal_flush:
                    current = queue_from_goal.popleft()
                    expand_frontier(
                        current, start, 
                        depth_from_goal, parent_from_goal, queue_from_goal, history_from_goal,
                        depth_from_start
                    )
            else:
                # 优先处理节点数较少的队列
                if queue_from_start and (not queue_from_goal or len(queue_from_start) <= len(queue_from_goal)):
                    current = queue_from_start.popleft()
                    expand_frontier(
                        current, goal, 
                        depth_from_start, parent_from_start, queue_from_start, history_from_start,
                        depth_from_goal
                    )
                elif queue_from_goal:
                    current = queue_from_goal.popleft()
                    expand_frontier(
                        current, start, 
                        depth_from_goal, parent_from_goal, queue_from_goal, history_from_goal,
                        depth_from_start
                    )
        
        # 路径重建
        if meet_point is None:
            return None, stats
        
        # 从相遇点回溯到起点
        path = []
        current = meet_point
        while current != start:
            path.append(current)
            current = parent_from_start[current[0]][current[1]]
        path.append(start)
        path.reverse()
        
        # 从相遇点回溯到终点
        current = meet_point
        while current != goal:
            current = parent_from_goal[current[0]][current[1]]
            path.append(current)
        
        return path, stats

# ----------------- Map loading functions -----------------
def load_map(path: str):
    grid = []
    S = E = None

    # 支持的符号集合
    WALL_CH = {CONFIG.WALL_U, CONFIG.WALL_A}
    EMPTY_CH = {CONFIG.EMPTY_U, CONFIG.EMPTY_A}
    START_CH = {CONFIG.START_U, CONFIG.START_A, "S", "s"}
    END_CH   = {CONFIG.END_U,   CONFIG.END_A,   "E", "e"}

    with open(path, "r", encoding="utf-8") as f:
        for r, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = []
            for c, ch in enumerate(line.replace(" ", "")):
                if ch in WALL_CH:
                    row.append(1)
                elif ch in EMPTY_CH:
                    row.append(0)
                elif ch in START_CH:
                    S = (r, c)
                    row.append(0)
                elif ch in END_CH:
                    E = (r, c)
                    row.append(0)
                else:
                    raise ValueError(f"bad char {repr(ch)} at {r},{c}")
            grid.append(row)

    if S is None or E is None:
        raise ValueError("map needs S/E")
    return grid, S, E

# ----------------- Random map generation -----------------
class LCG32:
    def __init__(self, seed: int): self.s = seed & 0xFFFFFFFF
    def next_u32(self) -> int:
        self.s = (1664525*self.s + 1013904223) & 0xFFFFFFFF
        return self.s
    def rand(self) -> float: return self.next_u32() / 2**32

def gen_random_map(w: int, h: int, p: float, seed: int):
    rng = LCG32(seed)
    g = [[0]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if rng.rand() < p:
                g[r][c] = 1
    g[0][0] = 0
    g[h-1][w-1] = 0
    return g, (0,0), (h-1,w-1)

# ----------------- Visualization -----------------
def visualize_grid(grid, path=None, start=None, goal=None):
    """可视化网格和路径，使用CONFIG中的符号配置"""
    if CONFIG.USE_ASCII:
        WALL = CONFIG.WALL_A
        EMPTY = CONFIG.EMPTY_A
        PATH = CONFIG.PATH_A
        START = CONFIG.START_A
        END = CONFIG.END_A
        ARROWS = CONFIG.ARROWS_A
    else:
        WALL = CONFIG.WALL_U
        EMPTY = CONFIG.EMPTY_U
        PATH = CONFIG.PATH_U
        START = CONFIG.START_U
        END = CONFIG.END_U
        ARROWS = CONFIG.ARROWS_U
    
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    # 创建路径箭头映射
    arrow_map = {}
    if CONFIG.ARROW_PATH and path and len(path) > 1:
        for i in range(1, len(path)):
            prev = path[i-1]
            curr = path[i]
            dr = curr[0] - prev[0]
            dc = curr[1] - prev[1]
            
            if dr == -1: arrow_map[curr] = ARROWS['U']
            elif dr == 1: arrow_map[curr] = ARROWS['D']
            elif dc == -1: arrow_map[curr] = ARROWS['L']
            elif dc == 1: arrow_map[curr] = ARROWS['R']
    
    # 渲染网格
    for r in range(height):
        line = []
        for c in range(width):
            pos = (r, c)
            
            if pos == start:
                line.append(START)
            elif pos == goal:
                line.append(END)
            elif path and pos in path:
                if pos in arrow_map and CONFIG.ARROW_PATH:
                    line.append(arrow_map[pos])
                else:
                    line.append(PATH)
            elif grid[r][c] == 1:
                line.append(WALL)
            else:
                line.append(EMPTY)
        print(''.join(line))

# ----------------- Path validation -----------------
def validate_path(grid, path, start, goal):
    """验证路径是否连续且有效"""
    if not path:
        return False
    
    # 检查起点和终点
    if path[0] != start or path[-1] != goal:
        return False
    
    # 检查路径连续性
    for i in range(1, len(path)):
        prev = path[i-1]
        curr = path[i]
        dr = abs(prev[0] - curr[0])
        dc = abs(prev[1] - curr[1])
        
        # 确保是相邻单元格
        if dr + dc != 1:
            return False
        
        # 确保单元格可通过
        if not (0 <= curr[0] < len(grid) and 0 <= curr[1] < len(grid[0])):
            return False
        if grid[curr[0]][curr[1]] != 0:
            return False
    
    return True

# ----------------- BFS for shortest path -----------------
def bfs_shortest_path(grid, start, goal):
    """BFS计算最短路径长度"""
    if not grid or not grid[0]:
        return -1
    
    height, width = len(grid), len(grid[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visited = [[False] * width for _ in range(height)]
    queue = deque([(start[0], start[1], 0)])
    visited[start[0]][start[1]] = True
    
    while queue:
        r, c, dist = queue.popleft()
        
        if (r, c) == goal:
            return dist
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < height and 0 <= nc < width and 
                not visited[nr][nc] and grid[nr][nc] == 0):
                visited[nr][nc] = True
                queue.append((nr, nc, dist + 1))
    
    return -1

# ----------------- Main function -----------------
def main():
    # 根据配置加载地图
    if CONFIG.USE_RANDOM:
        grid, start, goal = gen_random_map(
            CONFIG.RAND_W, CONFIG.RAND_H, 
            CONFIG.RAND_P, CONFIG.RAND_SEED
        )
        print(f"Generated random map: {CONFIG.RAND_W}x{CONFIG.RAND_H}, p={CONFIG.RAND_P}, seed={CONFIG.RAND_SEED}")
    else:
        if not CONFIG.MAP_FILE:
            raise ValueError("MAP_FILE must be specified when not using random map")
        grid, start, goal = load_map(CONFIG.MAP_FILE)
        print(f"Loaded map from: {CONFIG.MAP_FILE}")
    
    # 覆盖起点和终点（如果指定）
    if CONFIG.FILE_START:
        start = CONFIG.FILE_START
    if CONFIG.FILE_END:
        goal = CONFIG.FILE_END
    
    # 确保起点和终点可通过
    if grid[start[0]][start[1]] != 0 or grid[goal[0]][goal[1]] != 0:
        print("Warning: Start or goal position is blocked!")
    
    # 运行IBP-B*算法
    ibp = IBP_BStar(grid)
    path, stats = ibp.run_ibp_bstar(start, goal, CONFIG.WAIT_LAYERS)
    
    # 处理结果
    valid_path = False
    if path:
        valid_path = validate_path(grid, path, start, goal)
        if valid_path or CONFIG.PRINT_INVALID_PATH:
            print("\nPath found by IBP-B*:")
            visualize_grid(grid, path, start, goal)
        
        if CONFIG.PRINT_STATS:
            print(f"\nPath length: {len(path)}")
            print(f"Nodes expanded: {stats['expanded']}")
            print(f"Zigzag activations: {stats['zigzag_used']}")
            print(f"Path is {'valid' if valid_path else 'INVALID'}")
    else:
        print("\nNo path found by IBP-B*")
    
    # 计算BFS最短路径（用于比较）
    if CONFIG.PRINT_STATS:
        bfs_len = bfs_shortest_path(grid, start, goal)
        if bfs_len != -1:
            print(f"\nBFS shortest path length: {bfs_len}")
            if path and valid_path:
                efficiency = (len(path) / bfs_len - 1) * 100
                print(f"IBP-B* path is {efficiency:.2f}% longer than optimal")

# 运行主程序
if __name__ == "__main__":
    # 示例配置 - 可以根据需要修改
    CONFIG.USE_RANDOM = True
    CONFIG.RAND_W, CONFIG.RAND_H = 64, 64  # 较小的地图用于演示
    CONFIG.RAND_P = 0.2
    CONFIG.RAND_SEED = 15614646
    CONFIG.WAIT_LAYERS = 2
    CONFIG.PRINT_STATS = True
    CONFIG.PRINT_PATH = True
    CONFIG.ARROW_PATH = False
    CONFIG.USE_ASCII = True
    
    # 运行算法
    main()
