import torch
import numpy as np
from gomoku import GomokuGame
from nnet import GomokuNNet
from mcts import MCTS
from utils import load_checkpoint

def print_board(board):
    """簡單列印棋盤"""
    size = board.shape[0]
    for r in range(size):
        line = ""
        for c in range(size):
            cell = board[r, c]
            if cell == 1:
                line += "X "
            elif cell == -1:
                line += "O "
            else:
                line += ". "
        print(line)
    print()

def human_move(game):
    legal = game.get_legal_moves()
    while True:
        try:
            pos = input(f"請輸入落子位置（格式 row,col 從0開始，例如 2,3）: ")
            x, y = map(int, pos.strip().split(","))
            if (x, y) in legal:
                return (x, y)
            else:
                print("該位置不可落子，請重新輸入。")
        except:
            print("輸入格式錯誤，請依格式輸入。")

def play():
    board_size = 9
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    game = GomokuGame(board_size)
    nnet = GomokuNNet(board_size).to(device)
    load_checkpoint(nnet, "checkpoints/model_iter10.pt", device=device)  # 載入模型
    mcts = MCTS(game, nnet, {
        'num_mcts_sims': 50,
        'cpuct': 1.0,
        'board_size': board_size,
        'device': device
    })

    game.reset()
    print("遊戲開始！你先手用 X，AI 用 O。")
    human_player = 1
    ai_player = -1

    while True:
        print_board(game.board)
        if game.current_player == human_player:
            move = human_move(game)
            game.make_move(move)
        else:
            print("AI 思考中...")
            pi = mcts.get_action_probs(game, temp=0)
            move_idx = np.argmax(pi)
            move = (move_idx // board_size, move_idx % board_size)
            game.make_move(move)
            print(f"AI 落子：{move}")

        result = game.check_win()
        if result != 0:
            print_board(game.board)
            if result == human_player:
                print("恭喜！你贏了！🎉")
            else:
                print("AI 贏了，再接再厲！")
            break

        if game.is_draw():
            print_board(game.board)
            print("平手！")
            break

if __name__ == "__main__":
    play()
