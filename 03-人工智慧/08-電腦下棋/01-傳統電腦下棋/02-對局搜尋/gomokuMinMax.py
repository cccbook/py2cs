import pygame
import sys
import numpy as np

# 定義一些常量
WIDTH = 600
HEIGHT = 600
BOARD_SIZE = 15
LINE_WIDTH = 2
GRID_SIZE = WIDTH // (BOARD_SIZE + 1)  # 棋盤格子大小調整
SEARCH_DEPTH = 1
# 初始化 Pygame
pygame.init()

# 設置遊戲視窗
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('五子棋')

# 定義顏色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# 初始化棋盤
board = np.zeros((BOARD_SIZE, BOARD_SIZE))

# 繪製棋盤
def draw_board():
    screen.fill(WHITE)
    for i in range(1, BOARD_SIZE + 1):  # 修改這裡的範圍
        pygame.draw.line(screen, BLACK, (i * GRID_SIZE, GRID_SIZE), (i * GRID_SIZE, HEIGHT - GRID_SIZE), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (GRID_SIZE, i * GRID_SIZE), (WIDTH - GRID_SIZE, i * GRID_SIZE), LINE_WIDTH)

# 繪製棋子
def draw_pieces():
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 1:
                pygame.draw.circle(screen, BLACK, ((j + 1) * GRID_SIZE, (i + 1) * GRID_SIZE), GRID_SIZE // 2 - 2)
            elif board[i][j] == -1:
                pygame.draw.circle(screen, WHITE, ((j + 1) * GRID_SIZE, (i + 1) * GRID_SIZE), GRID_SIZE // 2 - 2)

# 檢查勝利條件
# 檢查勝利條件，並返回獲勝的玩家
def check_win():
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE - 4):
            if np.all(board[i, j:j+5] == 1):
                return 1  # 黑子獲勝
            if np.all(board[i, j:j+5] == -1):
                return -1  # 白子獲勝
            if np.all(board[j:j+5, i] == 1):
                return 1
            if np.all(board[j:j+5, i] == -1):
                return -1
    for i in range(BOARD_SIZE - 4):
        for j in range(BOARD_SIZE - 4):
            if np.all(np.diag(board[i:i+5, j:j+5]) == 1):
                return 1
            if np.all(np.diag(board[i:i+5, j:j+5]) == -1):
                return -1
            if np.all(np.diag(np.fliplr(board[i:i+5, j:j+5])) == 1):
                return 1
            if np.all(np.diag(np.fliplr(board[i:i+5, j:j+5])) == -1):
                return -1
    return 0  # 沒有玩家獲勝
def show_winner(winner):
    screen.fill(WHITE)  # Fill the screen with white color
    font = pygame.font.SysFont(None, 50)
    if winner == 1:
        text = font.render("Black wins!", True, BLACK)
    elif winner == -1:
        text = font.render("White wins!", True, BLACK)  # Change the color to black
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)

    # Draw options with borders
    font_menu = pygame.font.SysFont(None, 40)
    text_menu = font_menu.render("Back to Main Menu", True, BLACK)
    text_rect_menu = text_menu.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))

    # Draw rectangles with borders around the buttons
    menu_rect = pygame.Rect(text_rect_menu.x - 5, text_rect_menu.y - 5, text_rect_menu.width + 10, text_rect_menu.height + 10)
    pygame.draw.rect(screen, BLACK, menu_rect, 2)

    screen.blit(text_menu, text_rect_menu)

    pygame.display.update()

    # Draw options
    font_menu = pygame.font.SysFont(None, 40)
    text_menu = font_menu.render("Back to Main Menu", True, BLACK)
    text_rect_menu = text_menu.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
    screen.blit(text_menu, text_rect_menu)


    pygame.display.update()

def choose_mode():
    screen.fill(WHITE)
    font = pygame.font.SysFont(None, 50)
    text_pvp = font.render("PVP", True, BLACK)
    text_pve = font.render("PVE", True, BLACK)
    text_rect_pvp = text_pvp.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
    text_rect_pve = text_pve.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
    screen.blit(text_pvp, text_rect_pvp)
    screen.blit(text_pve, text_rect_pve)
    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if text_rect_pvp.collidepoint(x, y):
                    return "PVP"
                elif text_rect_pve.collidepoint(x, y):
                    return "PVE"

def main():
    while True:
        mode = choose_mode()
        if mode == "PVP":
            result = pvp_game()
            if result == "MENU":
                continue  # 返回主菜单
        elif mode == "PVE":
            result = pve_game()
            if result == "MENU":
                continue  # 返回主菜单


def pvp_game():
    global board  # Declare board variable as global
    player = 1  # Black starts
    game_over = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                x, y = event.pos
                col = round(x / GRID_SIZE) - 1  # Calculate column index, rounded to nearest grid intersection
                row = round(y / GRID_SIZE) - 1  # Calculate row index, rounded to nearest grid intersection

                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] == 0:  # Check if within the board
                    board[row][col] = player
                    winner = check_win()
                    player *= -1
                    if winner != 0:
                        game_over = True

        if not game_over:
            draw_board()
            draw_pieces()
        pygame.display.update()

        if game_over:
            show_winner(winner)
            while True:  # Enter a new loop after the game ends, waiting for player's choice
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        if 200 <= x <= 400 and 300 <= y <= 350:  # Click "Back to Main Menu"
                            board = np.zeros((BOARD_SIZE, BOARD_SIZE))  # Clear the board
                            return "MENU"

        if not game_over:
            draw_board()
            draw_pieces()
        pygame.display.update()

def pve_game():
    global board  # 声明棋盘变量为全局变量
    player = 1  # 黑子先手
    game_over = False

    while True:
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        """
        # 黑子玩家下子
        if player == 1:
            # 极大极小值搜索，获取最佳下子位置
            best_move = get_best_move(board, SEARCH_DEPTH)
            make_move(board, best_move, player)
            player *= -1  # 切换到白子玩家
            # 检查游戏是否结束
            if check_win() != 0:
                game_over = True

        # 白子玩家下子（玩家）
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # print('event=', event)
                if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                    x, y = event.pos
                    col = round(x / GRID_SIZE) - 1  # 计算列索引，四舍五入到最近的格子交叉点
                    row = round(y / GRID_SIZE) - 1  # 计算行索引，四舍五入到最近的格子交叉点

                    # 检查是否在棋盘内，并且当前位置为空
                    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] == 0:
                        board[row][col] = player
                        player *= -1  # 切换到黑子玩家
                        # 检查游戏是否结束
                        if check_win() != 0:
                            game_over = True

        # 绘制棋盘和棋子
        draw_board()
        draw_pieces()
        pygame.display.update()

        # 如果游戏结束，显示获胜信息并等待玩家操作
        if game_over:
            show_winner(check_win())
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        if 200 <= x <= 400 and 300 <= y <= 350:  # 点击“返回主菜单”
                            board = np.zeros((BOARD_SIZE, BOARD_SIZE))  # 清空棋盘
                            return

# 获取最佳下子位置
def get_best_move(board, depth):
    best_score = float("-inf")
    best_move = None
    if depth <= 0:  # 如果達到了最大深度，則直接返回
        return best_move
    for move in possible_moves(board):
        make_move(board, move, 1)
        score = minimax(board, depth - 1, False, move)
        # if score > 0: print('move=', move, 'scroe=', score)
        undo_move(board, move)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


# 评估棋盘得分
def evaluate_board(board, move):
    score = 0
    for i in range(len(board) - 4):
        for j in range(len(board) - 4):
            matrix = board[i:i+5, j:j+5]
            for row in matrix:
                score += evaluate_line(row)
            for col in matrix.T:
                score += evaluate_line(col)
            score += evaluate_line(np.diag(matrix))
            score += evaluate_line(np.diag(np.fliplr(matrix)))
    # 盡量下中間
    x,y = move
    score -= 0.01*abs(x-7) + 0.01*abs(y-7)
    return score

# 评估一行的得分
def evaluate_line(line):
    score = 0
    length = len(line)
    i = 0
    while i < length:
        if line[i] == 0:
            i += 1
            continue
        start = i
        count = 0
        while i < length and line[i] == line[start]:
            count += 1
            i += 1
        if count >= 5:
            return float("inf") if line[start] == 1 else float("-inf")
        if count == 4:
            score += 100 if line[start] == 1 else -100
        elif count == 3:
            score += 10 if line[start] == 1 else -10
        elif count == 2:
            score += 1 if line[start] == 1 else -1
    return score

# 极大极小值搜索
def minimax(board, depth, maximizing_player, move):
    if depth == 0 or check_win() != 0:
        return evaluate_board(board, move)
    
    if maximizing_player:
        max_score = float("-inf")
        for move in possible_moves(board):
            make_move(board, move, 1)
            score = minimax(board, depth - 1, False, move)
            undo_move(board, move)
            max_score = max(max_score, score)
        return max_score
    else:
        min_score = float("inf")
        for move in possible_moves(board):
            make_move(board, move, -1)
            score = minimax(board, depth - 1, True, move)
            undo_move(board, move)
            min_score = min(min_score, score)
        return min_score

# 获取所有可能的下棋位置
def possible_moves(board):
    moves = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                moves.append((i, j))
    return moves

# 下子
def make_move(board, move, player):
    board[move[0]][move[1]] = player

# 撤销下子
def undo_move(board, move):
    board[move[0]][move[1]] = 0

if __name__ == '__main__':
    main()

