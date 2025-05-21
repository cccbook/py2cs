'''
五子棋遊戲 - 命令列版本 (使用 Alpha-Beta Pruning)

使用方法：
人對人下  ：python gomoku_ab.py P P
人對電腦  ：python gomoku_ab.py P C
電腦對電腦：python gomoku_ab.py C C
'''

import sys
import time
import random
import copy

# Alpha-Beta 搜索的最大深度
MAX_DEPTH = 3
# 搜索範圍 (棋子周圍多少格內)
SEARCH_RADIUS = 2

class Board:
    def __init__(self, size=15):
        self.size = size
        self.grid = [['-' for _ in range(size)] for _ in range(size)]
        self.last_move = None
        self.moves_count = 0
    
    def display(self):
        """顯示棋盤"""
        # 顯示列標籤
        print('  ' + ' '.join([hex(i)[2:] for i in range(self.size)]))
        # 顯示棋盤內容和行標籤
        for i in range(self.size):
            print(hex(i)[2:] + ' ' + ' '.join(self.grid[i]) + ' ' + hex(i)[2:])
        # 再顯示一次列標籤
        print('  ' + ' '.join([hex(i)[2:] for i in range(self.size)]))
    
    def is_valid_move(self, row, col):
        """檢查移動是否有效"""
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        return self.grid[row][col] == '-'
    
    def make_move(self, row, col, player):
        """下子"""
        if self.is_valid_move(row, col):
            self.grid[row][col] = player
            self.last_move = (row, col)
            self.moves_count += 1
            return True
        return False
    
    def undo_move(self, row, col):
        """撤銷移動"""
        if 0 <= row < self.size and 0 <= col < self.size and self.grid[row][col] != '-':
            self.grid[row][col] = '-'
            self.moves_count -= 1
            return True
        return False
    
    def is_full(self):
        """檢查棋盤是否已滿"""
        return self.moves_count >= self.size * self.size
    
    def check_win(self, row, col, player):
        """檢查在指定位置下子後是否獲勝"""
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        
        if self.grid[row][col] != player:
            return False
            
        # 檢查方向: 水平、垂直、對角線和反對角線
        directions = [
            [(0, 1), (0, -1)],   # 水平
            [(1, 0), (-1, 0)],   # 垂直
            [(1, 1), (-1, -1)],  # 主對角線
            [(1, -1), (-1, 1)]   # 副對角線
        ]
        
        for dirs in directions:
            count = 1  # 中心點算一個
            
            for dx, dy in dirs:
                # 沿著方向檢查連續的棋子
                for step in range(1, 5):  # 最多檢查4步 (加上中心點共5個)
                    nx, ny = row + dx * step, col + dy * step
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[nx][ny] == player:
                        count += 1
                    else:
                        break
            
            if count >= 5:  # 如果有連續5個或以上，獲勝
                return True
                
        return False
    
    def get_neighbors(self, radius=SEARCH_RADIUS):
        """獲取已下棋子周圍的空位"""
        neighbors = set()
        
        if self.moves_count == 0:  # 如果棋盤為空，返回中心點
            center = self.size // 2
            return [(center, center)]
            
        # 檢查所有已下棋子周圍的空位
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] != '-':
                    # 檢查這個棋子周圍的位置
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < self.size and 0 <= nc < self.size and 
                                self.grid[nr][nc] == '-' and (dr != 0 or dc != 0)):
                                neighbors.add((nr, nc))
        
        # 如果沒有找到鄰居（例如棋盤已滿），返回空列表
        return list(neighbors) if neighbors else []
    
    def evaluate(self):
        """評估當前棋盤狀態的分數"""
        # 正分代表對 'x' 有利，負分代表對 'o' 有利
        score_x = self._evaluate_for('x')
        score_o = self._evaluate_for('o')
        return score_x - score_o
    
    def _evaluate_for(self, player):
        """計算某一玩家的棋盤評分"""
        score = 0
        opponent = 'o' if player == 'x' else 'x'
        
        # 檢查所有方向的棋型
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == player:
                    # 水平方向
                    score += self._evaluate_direction(r, c, 0, 1, player)
                    # 垂直方向
                    score += self._evaluate_direction(r, c, 1, 0, player)
                    # 主對角線方向
                    score += self._evaluate_direction(r, c, 1, 1, player)
                    # 副對角線方向
                    score += self._evaluate_direction(r, c, 1, -1, player)
        
        return score
    
    def _evaluate_direction(self, row, col, dr, dc, player):
        """評估從(row,col)開始，在(dr,dc)方向上的棋型"""
        consecutive = 1  # 連續的棋子數
        blocked = 0      # 是否被阻擋
        score = 0        # 初始分數
        
        # 正向檢查
        r, c = row + dr, col + dc
        while 0 <= r < self.size and 0 <= c < self.size:
            if self.grid[r][c] == player:
                consecutive += 1
            elif self.grid[r][c] == '-':
                break
            else:  # 對手的棋子
                blocked += 1
                break
            r += dr
            c += dc
        
        # 反向檢查
        r, c = row - dr, col - dc
        while 0 <= r < self.size and 0 <= c < self.size:
            if self.grid[r][c] == player:
                consecutive += 1
            elif self.grid[r][c] == '-':
                break
            else:  # 對手的棋子
                blocked += 1
                break
            r -= dr
            c -= dc
        
        # 根據連續棋子數和阻擋情況評分
        if consecutive >= 5:
            score = 100000  # 五連獲勝
        elif consecutive == 4:
            if blocked == 0:
                score = 10000  # 活四
            elif blocked == 1:
                score = 1000   # 死四
        elif consecutive == 3:
            if blocked == 0:
                score = 1000   # 活三
            elif blocked == 1:
                score = 100    # 死三
        elif consecutive == 2:
            if blocked == 0:
                score = 100    # 活二
            elif blocked == 1:
                score = 10     # 死二
        elif consecutive == 1:
            score = 1
        
        return score

def alpha_beta(board, depth, alpha, beta, maximizing_player, player):
    """
    Alpha-Beta 修剪算法
    
    參數:
    - board: 棋盤對象
    - depth: 當前搜索深度
    - alpha: Alpha 值
    - beta: Beta 值
    - maximizing_player: 是否為最大化玩家
    - player: 當前玩家 ('x' 或 'o')
    
    返回:
    - 最佳得分和最佳移動 (score, (row, col))
    """
    opponent = 'o' if player == 'x' else 'x'
    
    # 檢查終止條件
    if board.last_move is not None:
        last_r, last_c = board.last_move
        if board.check_win(last_r, last_c, player):
            return 100000, None  # 當前玩家獲勝
        if board.check_win(last_r, last_c, opponent):
            return -100000, None  # 對手獲勝
    
    if depth == 0 or board.is_full():
        return board.evaluate() if player == 'x' else -board.evaluate(), None
    
    # 獲取可能的移動
    valid_moves = board.get_neighbors()
    if not valid_moves:
        return 0, None  # 平局或無處可走
    
    # 隨機排序移動以增加剪枝效率
    random.shuffle(valid_moves)
    
    best_move = None
    
    if maximizing_player:  # 最大化玩家 (x)
        max_eval = float('-inf')
        for move in valid_moves:
            r, c = move
            if board.make_move(r, c, player):
                eval_score, _ = alpha_beta(board, depth - 1, alpha, beta, False, opponent)
                board.undo_move(r, c)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta 剪枝
        
        return max_eval, best_move
    else:  # 最小化玩家 (o)
        min_eval = float('inf')
        for move in valid_moves:
            r, c = move
            if board.make_move(r, c, player):
                eval_score, _ = alpha_beta(board, depth - 1, alpha, beta, True, opponent)
                board.undo_move(r, c)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha 剪枝
        
        return min_eval, best_move

def human_turn(board, player):
    """人類玩家的回合"""
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
    """電腦玩家的回合"""
    print(f'電腦({player})正在思考...')
    start_time = time.time()
    
    # 第一步棋直接下在中心點
    if board.moves_count == 0:
        center = board.size // 2
        board.make_move(center, center, player)
        print(f'電腦({player})下在: {hex(center)[2:]}{hex(center)[2:]}')
        return center, center
    
    # 使用Alpha-Beta搜索找最佳移動
    score, move = alpha_beta(board, MAX_DEPTH, float('-inf'), float('inf'), 
                             player == 'x', player)
    
    if move:
        row, col = move
        board.make_move(row, col, player)
        end_time = time.time()
        print(f'電腦({player})思考了 {end_time - start_time:.2f} 秒')
        print(f'電腦({player})下在: {hex(row)[2:]}{hex(col)[2:]}')
        return row, col
    else:
        # 備用方案：如果Alpha-Beta沒找到好的移動，隨機選一個合法位置
        valid_moves = [(r, c) for r in range(board.size) for c in range(board.size) 
                       if board.grid[r][c] == '-']
        if valid_moves:
            row, col = random.choice(valid_moves)
            board.make_move(row, col, player)
            print(f'電腦({player})下在: {hex(row)[2:]}{hex(col)[2:]}')
            return row, col
        return None, None  # 棋盤已滿

def play_game(player_o, player_x):
    """開始遊戲"""
    board = Board()
    board.display()
    
    current_player = 'o'  # 'o' 先手
    
    while True:
        # 決定誰來下棋
        if current_player == 'o':
            play_type = player_o
        else:
            play_type = player_x
            
        # 執行下棋動作
        if play_type.upper() == 'P':
            row, col = human_turn(board, current_player)
        else:
            row, col = computer_turn(board, current_player)
            time.sleep(0.5)  # 電腦下棋後稍微暫停一下，讓人類可以看清
        
        # 顯示棋盤
        print("\n當前棋局：")
        board.display()
        
        # 檢查勝負
        if board.check_win(row, col, current_player):
            print(f'\n{current_player} 獲勝！')
            break
            
        # 檢查是否平局
        if board.is_full():
            print('\n平局！')
            break
            
        # 切換玩家
        current_player = 'x' if current_player == 'o' else 'o'

if __name__ == "__main__":
    # 處理命令行參數
    if len(sys.argv) < 3:
        print('使用方式: python gomoku_ab.py [o玩家類型] [x玩家類型]')
        print('玩家類型: P=人類, C=電腦')
        print('預設使用: P C (人類對電腦)')
        player_o, player_x = 'P', 'C'
    else:
        player_o, player_x = sys.argv[1], sys.argv[2]
    
    # 開始遊戲
    play_game(player_o, player_x)