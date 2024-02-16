# https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
An example implementation of the abstract Node class for use in MCTS
If you run this file then you can play against the computer.
A tic-tac-toe board is represented as a tuple of 9 values, each either None,
True, or False, respectively meaning 'empty', 'X', and 'O'.
The board is indexed by row:
0 1 2
3 4 5
6 7 8
For example, this game board
O - X
O X -
X - -
corrresponds to this tuple:
(False, None, True, False, True, None, True, None, None)
"""

"""
一個在 MCTS 中使用的抽象 Node 類別的範例實作
如果你執行這個檔案，你可以與電腦玩井字遊戲
井字遊戲的棋盤以一個有 9 個值的元組表示，每個值為 None、True 或 False，分別代表 '空'、'X' 和 'O'
棋盤是以列為索引:
0 1 2
3 4 5
6 7 8
例如，這個遊戲棋盤
O - X
O X -
X - -
對應到這個元組:
(False, None, True, False, True, None, True, None, None)
"""

from collections import namedtuple
from random import choice
from mcts import MCTS, Node

# 用 nametuple 創建類別，包含 tup, turn, winner, terminal 等函數。
_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")

# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
# 繼承自 namedtuple 很方便，因為它使類別成為不可變的，
# 並預定義了 __init__、__repr__、__hash__、__eq__ 等方法

class TicTacToeBoard(_TTTB, Node): # 多重繼承 (_TTTB+Node)
    def find_children(board):
        if board.terminal:  # 如果遊戲結束則無法進行任何移動 If the game is finished then no moves can be made 
            return set()
        # 否則，可以在每個空位上進行一步移動 Otherwise, you can make a move in each of the empty spots 
        return {
            board.make_move(i) for i, value in enumerate(board.tup) if value is None
        }

    def find_random_child(board):
        if board.terminal:
            return None  # 如果遊戲結束則無法進行任何移動 If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(board.tup) if value is None] # 找一個空格隨機下子
        return board.make_move(choice(empty_spots)) # 傳回下子後的盤面

    def reward(board): # 取得報酬
        if not board.terminal: # 只有遊戲結束，才能取得回饋 reward
            raise RuntimeError(f"reward called on nonterminal board {board}")
        if board.winner is board.turn:
            # 輪到你，但你已經獲勝。這應該是不可能的。 It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board {board}")
        if board.turn is (not board.winner):
            return 0  # 對手剛獲勝。不好。 Your opponent has just won. Bad.
        if board.winner is None:
            return 0.5  # 平局 Board is a tie
        # 獲勝者既不是 True、False、也不是 None (錯誤) The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {board.winner}")

    def is_terminal(board): # 是否遊戲結束了
        return board.terminal

    def make_move(board, index): # 下子在 index 後更新盤面
        tup = board.tup[:index] + (board.turn,) + board.tup[index + 1 :] # tup = 下子之後的盤面
        turn = not board.turn # 角色互換
        winner = _find_winner(tup) # True 是 X 贏，False 是 O 贏，None 才是沒人贏
        is_terminal = (winner is not None) or not any(v is None for v in tup) # 遊戲是否結束
        return TicTacToeBoard(tup, turn, winner, is_terminal)

    def to_pretty_string(board): # 轉成格式好看的盤面字串
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(board.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )


def play_game():
    tree = MCTS()
    board = new_tic_tac_toe_board() # 創建新盤面
    print(board.to_pretty_string()) # 印出
    while True:
        # 玩家下子 (含處理錯誤)
        row_col = input("enter row,col: ") # 玩家輸入下子位置
        try:
            row, col = map(int, row_col.split(","))
            index = 3 * (row - 1) + (col - 1)
            if board.tup[index] is not None:
                raise RuntimeError("Invalid move")
        except Exception as err:
            print(err)
            continue
        board = board.make_move(index) # 下子後更新盤面
        print(board.to_pretty_string()) # 印出盤面
        if board.terminal: # 如果遊戲結束，離開
            break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        # 您可以一邊進行訓練，一邊進行遊戲，或者只在開始時進行訓練。
        # 在這裡，我們在每個回合中進行 50 次模擬。
        for _ in range(50):
            tree.do_rollout(board)
        board = tree.choose(board) # AI 下子 (用 MCTS 搜尋樹找下法，這行是關鍵)
        print(board.to_pretty_string())
        if board.terminal: # 如果遊戲結束，離開
            break


def _winning_combos():
    for start in range(0, 9, 3):  # 3 子連成一列 three in a row 
        yield (start, start + 1, start + 2)
    for start in range(3):  # 3 子連成一行 three in a column 
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # 下斜線 down-right diagonal
    yield (2, 4, 6)  # 上斜線 down-left diagonal


def _find_winner(tup):
    "Returns None if no winner, True if X wins, False if O wins"
    "傳回輸贏結果 (贏家)"
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if False is v1 is v2 is v3:
            return False
        if True is v1 is v2 is v3:
            return True
    return None


def new_tic_tac_toe_board():
    # tup=(None,) * 9: 一開始盤面是空的
    # turn=True: 換玩家下了
    # winner=None: 沒有人贏了
    # terminal=False: 遊戲還沒結束
    return TicTacToeBoard(tup=(None,) * 9, turn=True, winner=None, terminal=False)


if __name__ == "__main__":
    play_game() # 啟動遊戲