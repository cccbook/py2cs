from gomoku import GomokuGame
from nnet import GomokuNNet
from mcts import MCTS
from coach import Coach
import torch

args = {
    'board_size': 9,
    'num_iters': 10,
    'num_eps': 20,
    'num_mcts_sims': 50,
    'cpuct': 1.0,
    'lr': 1e-3,
    'epochs': 5,
    'batch_size': 32,
    'max_examples': 10000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

if __name__ == "__main__":
    coach = Coach(GomokuGame, GomokuNNet, MCTS, args)
    coach.learn()
