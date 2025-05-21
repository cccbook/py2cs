# coach.py
import random
import numpy as np
import torch
from tqdm import tqdm
from utils import save_checkpoint

class Coach:
    def __init__(self, game_class, nnet_class, mcts_class, args):
        self.args = args
        self.game = game_class(args['board_size'])
        self.nnet = nnet_class(args['board_size']).to(args['device'])
        self.mcts = mcts_class(self.game, self.nnet, args)
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=args['lr'])
        self.train_examples = []

    def execute_episode(self):
        train_examples = []
        self.game.reset()
        states, mcts_probs, players = [], [], []

        while True:
            canonical_board = self.game.get_canonical_board()
            pi = self.mcts.get_action_probs(self.game, temp=1)
            move = np.random.choice(len(pi), p=pi)
            x, y = divmod(move, self.args['board_size'])

            states.append(np.copy(canonical_board))
            mcts_probs.append(pi)
            players.append(self.game.current_player)

            self.game.make_move((x, y))
            result = self.game.check_win()
            if result != 0:
                return [(s, p, result * player) for s, p, player in zip(states, mcts_probs, players)]
            if self.game.is_draw():
                return [(s, p, 0) for s, p, player in zip(states, mcts_probs, players)]

    def learn(self):
        for iteration in range(1, self.args['num_iters'] + 1):
            print(f"\nSelf-play iteration {iteration}")
            iteration_examples = []
            for _ in tqdm(range(self.args['num_eps'])):
                iteration_examples += self.execute_episode()

            self.train_examples += iteration_examples
            if len(self.train_examples) > self.args['max_examples']:
                self.train_examples = self.train_examples[-self.args['max_examples']:]

            self.train()

            save_checkpoint(self.nnet, f"checkpoints/model_iter{iteration}.pt")


    def train(self):
        print("Training on examples:", len(self.train_examples))
        batch_size = self.args['batch_size']
        for epoch in range(self.args['epochs']):
            print(f"Epoch {epoch+1}/{self.args['epochs']}")
            random.shuffle(self.train_examples)
            for i in range(0, len(self.train_examples), batch_size):
                sample = self.train_examples[i:i+batch_size]
                boards, pis, zs = zip(*sample)
                boards = torch.tensor(boards, dtype=torch.float32).unsqueeze(1).to(self.args['device'])
                pis = torch.tensor(pis, dtype=torch.float32).to(self.args['device'])
                zs = torch.tensor(zs, dtype=torch.float32).unsqueeze(1).to(self.args['device'])

                out_pi, out_v = self.nnet(boards)
                loss_pi = -torch.sum(pis * out_pi) / len(pis)
                loss_v = torch.mean((zs - out_v) ** 2)
                total_loss = loss_pi + loss_v

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
