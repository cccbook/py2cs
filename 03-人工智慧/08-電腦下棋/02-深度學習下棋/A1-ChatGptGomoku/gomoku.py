import numpy as np
import random

class GomokuEnv:
    def __init__(self, board_size=16, win_length=5):
        self.board_size = board_size
        self.win_length = win_length
        self.reset()
        
    def reset(self):
        """ 重置棋盘 """
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.done = False
        self.winner = None
        self.current_player = 1
        return self.board
    
    def is_valid_action(self, action):
        """ 判断该位置是否有效 """
        x, y = action
        return self.board[x, y] == 0
    
    def step(self, action):
        """ 执行一步棋，并返回新的状态和奖励 """
        x, y = action
        if not self.is_valid_action(action):
            raise ValueError(f"Invalid action {action}")
        
        self.board[x, y] = self.current_player
        
        # 检查是否有玩家获胜
        if self.check_win(x, y):
            self.done = True
            self.winner = self.current_player
            reward = 1
        elif not self.is_board_full():
            self.current_player = -self.current_player  # 轮到另一位玩家
            reward = 0
        else:
            self.done = True
            reward = 0.5  # 平局
            
        return self.board, reward, self.done, {}
    
    def is_board_full(self):
        """ 棋盘是否已满 """
        return np.all(self.board != 0)
    
    def check_win(self, x, y):
        """ 检查当前玩家是否获胜 """
        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for i in range(1, self.win_length):
                if 0 <= x + i * dx < self.board_size and 0 <= y + i * dy < self.board_size:
                    if self.board[x + i * dx, y + i * dy] == player:
                        count += 1
                    else:
                        break
            for i in range(1, self.win_length):
                if 0 <= x - i * dx < self.board_size and 0 <= y - i * dy < self.board_size:
                    if self.board[x - i * dx, y - i * dy] == player:
                        count += 1
                    else:
                        break
            if count >= self.win_length:
                return True
        return False
    
    def render(self):
        """ 显示棋盘 """
        print(self.board)

