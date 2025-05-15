'''
簡化的五子棋遊戲 - 命令列版本

使用方法：
人對人下  ：python gomoku_claude.py P P
人對電腦  ：python gomoku_claude.py P C
電腦對電腦：python gomoku_claude.py C C
'''

import sys
import time
import random

class Board:
    def __init__(self, size=15):
        # 初始化棋盤大小，預設為15×15
        self.size = size
        # 建立二維陣列作為棋盤
        self.grid = [['-' for _ in range(size)] for _ in range(size)]
    
    def display(self):
        # 顯示棋盤
        # 顯示行座標
        print('  ' + ' '.join([hex(i)[2:] for i in range(self.size)]))
        # 顯示每一行及列座標
        for i in range(self.size):
            print(hex(i)[2:] + ' ' + ' '.join(self.grid[i]) + ' ' + hex(i)[2:])
        # 顯示行座標
        print('  ' + ' '.join([hex(i)[2:] for i in range(self.size)]))
    
    def is_valid_move(self, row, col):
        # 檢查移動是否有效
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if self.grid[row][col] != '-':
            return False
        return True
    
    def make_move(self, row, col, player):
        # 下子
        if self.is_valid_move(row, col):
            self.grid[row][col] = player
            return True
        return False
    
    def check_win(self, row, col, player):
        # 檢查勝負的方向：水平、垂直、主對角線、副對角線
        directions = [
            [(0, 1), (0, -1)],  # 水平
            [(1, 0), (-1, 0)],  # 垂直
            [(1, 1), (-1, -1)], # 主對角線
            [(1, -1), (-1, 1)]  # 副對角線
        ]
        
        for direction in directions:
            count = 1  # 包含目前位置
            
            # 向兩個方向檢查
            for dx, dy in direction:
                for step in range(1, 5):  # 五子棋需要連續5顆
                    r, c = row + dx * step, col + dy * step
                    if 0 <= r < self.size and 0 <= c < self.size and self.grid[r][c] == player:
                        count += 1
                    else:
                        break
            
            if count >= 5:
                return True
        
        return False
    
    def is_full(self):
        # 檢查棋盤是否已滿
        for row in self.grid:
            if '-' in row:
                return False
        return True

def human_turn(board, player):
    while True:
        try:
            move = input(f'請輸入{player}的位置 (例如: 88): ')
            if len(move) != 2:
                print('請輸入兩個字元，分別代表列和行')
                continue
                
            row = int(move[0], 16)
            col = int(move[1], 16)
            
            if board.make_move(row, col, player):
                return row, col
            else:
                print('無效的移動，請重試')
        except ValueError:
            print('請輸入有效的十六進制數字')

def computer_turn(board, player):
    # 對手的棋子
    opponent = 'o' if player == 'x' else 'x'
    
    # 評分函數 - 計算每個位置的分數
    def score_position(r, c):
        if not board.is_valid_move(r, c):
            return -1
        
        score = 0
        
        # 檢查各方向
        directions = [
            [(0, 1), (0, -1)],  # 水平
            [(1, 0), (-1, 0)],  # 垂直
            [(1, 1), (-1, -1)], # 主對角線
            [(1, -1), (-1, 1)]  # 副對角線
        ]
        
        # 攻擊分數權重比防守略高
        for direction in directions:
            # 計算攻擊分數
            score += check_line(r, c, direction, player, board)
            # 計算防守分數，權重略低
            score += check_line(r, c, direction, opponent, board) * 0.9
        
        # 中心位置優先
        center = board.size // 2
        distance_from_center = abs(r - center) + abs(c - center)
        score -= distance_from_center * 0.1
        
        return score
    
    # 檢查某方向上的連子情況
    def check_line(r, c, direction, piece, board):
        line_score = 0
        empty_ends = 0  # 記錄兩端是否為空
        consecutive = 0  # 連續棋子數
        blocked = 0  # 是否被阻擋
        
        # 暫時放置棋子以評估
        original = board.grid[r][c]
        board.grid[r][c] = piece
        
        for dx, dy in direction:
            blocked_end = False
            for step in range(1, 5):
                nr, nc = r + dx * step, c + dy * step
                if 0 <= nr < board.size and 0 <= nc < board.size:
                    if board.grid[nr][nc] == piece:
                        consecutive += 1
                    elif board.grid[nr][nc] == '-':
                        empty_ends += 1
                        break
                    else:
                        blocked_end = True
                        break
                else:
                    blocked_end = True
                    break
            
            if blocked_end:
                blocked += 1
        
        # 恢復原來的位置
        board.grid[r][c] = original
        
        # 基於連續棋子數和兩端情況評分
        if consecutive == 4:
            line_score = 1000  # 必勝
        elif consecutive == 3:
            if empty_ends == 2:
                line_score = 100  # 活三
            else:
                line_score = 10  # 死三
        elif consecutive == 2:
            if empty_ends == 2:
                line_score = 5   # 活二
            else:
                line_score = 3   # 死二
        elif consecutive == 1:
            line_score = 1
        
        # 如果兩端都被堵住，分數大幅降低
        if blocked == 2:
            line_score /= 4
            
        return line_score
    
    # 找出最佳位置
    best_score = -float('inf')
    best_positions = []
    
    for r in range(board.size):
        for c in range(board.size):
            if board.is_valid_move(r, c):
                pos_score = score_position(r, c)
                if pos_score > best_score:
                    best_score = pos_score
                    best_positions = [(r, c)]
                elif pos_score == best_score:
                    best_positions.append((r, c))
    
    # 如果棋盤為空，優先下在中心位置
    if all(board.grid[r][c] == '-' for r in range(board.size) for c in range(board.size)):
        center = board.size // 2
        row, col = center, center
    else:
        # 從最佳位置中隨機選擇一個
        row, col = random.choice(best_positions) if best_positions else (board.size // 2, board.size // 2)
    
    board.make_move(row, col, player)
    print(f'電腦({player})下在: {hex(row)[2:]}{hex(col)[2:]}')
    return row, col

def play_game(player_o, player_x):
    board = Board()
    board.display()
    
    last_move = None
    
    while True:
        # 玩家 o 的回合
        if player_o.upper() == 'P':
            row, col = human_turn(board, 'o')
        else:
            row, col = computer_turn(board, 'o')
            time.sleep(1)  # 讓電腦思考看起來更真實
        
        board.display()
        
        # 檢查是否獲勝
        if board.check_win(row, col, 'o'):
            print('o 獲勝！')
            break
        
        # 檢查是否平局
        if board.is_full():
            print('平局！')
            break
        
        # 玩家 x 的回合
        if player_x.upper() == 'P':
            row, col = human_turn(board, 'x')
        else:
            row, col = computer_turn(board, 'x')
            time.sleep(1)  # 讓電腦思考看起來更真實
        
        board.display()
        
        # 檢查是否獲勝
        if board.check_win(row, col, 'x'):
            print('x 獲勝！')
            break
        
        # 檢查是否平局
        if board.is_full():
            print('平局！')
            break

if __name__ == "__main__":
    # 如果沒有足夠的命令行參數，給出預設值
    if len(sys.argv) < 3:
        print('使用方式: python gomoku_claude.py [o玩家類型] [x玩家類型]')
        print('玩家類型: P=人類, C=電腦')
        print('預設使用: P C (人類對電腦)')
        player_o, player_x = 'P', 'C'
    else:
        player_o, player_x = sys.argv[1], sys.argv[2]
    
    play_game(player_o, player_x)