import turtle

# 繪製棋盤
def draw_board(size, line_color, fill_color1, fill_color2):
    turtle.speed(0)
    turtle.pencolor(line_color)
    turtle.fillcolor(fill_color1)
    turtle.begin_fill()
    for i in range(4):
        turtle.forward(size * 8)
        turtle.right(90)
    turtle.end_fill()
    for i in range(10):
        for j in range(8):
            turtle.penup()
            turtle.goto(size * (j - 4), size * (4 - i))
            turtle.pendown()
            turtle.begin_fill()
            turtle.fillcolor(fill_color1 if (i + j) % 2 else fill_color2)
            for k in range(4):
                turtle.forward(size)
                turtle.right(90)
            turtle.end_fill()

# 繪製棋子
def draw_piece(x, y, color):
    turtle.penup()
    turtle.goto(x, y)
    turtle.dot(40, color)

# DFS 搜索
def dfs(board, color, depth):
    if depth == 0:
        return board
    max_board = None
    max_score = -float('inf')
    for i in range(10):
        for j in range(8):
            if board[i][j] == 0:
                board[i][j] = color
                score = eval_board(board)
                if score > max_score:
                    max_board = board.copy()
                    max_score = score
                board[i][j] = 0
    return dfs(max_board, -color, depth - 1)

# 評估棋盤
def eval_board(board):
    score = 0
    for i in range(10):
        for j in range(8):
            if board[i][j] == 0:
                continue
            for dx, dy in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
                x, y = i, j
                cnt = 0
                while 0 <= x < 10 and 0 <= y <
