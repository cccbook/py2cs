# mcts.py
import numpy as np
import math
import copy
import torch

class MCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # Q(s,a)
        self.Nsa = {}  # N(s,a)
        self.Ns = {}   # N(s)
        self.Ps = {}   # P(s)

        self.Es = {}   # 結果快取（+1 / -1 / 0）
        self.Vs = {}   # 合法動作

    def search(self, state):
        s_key = self._string_rep(state.board, state.current_player)

        if s_key not in self.Es:
            self.Es[s_key] = state.check_win()
        if self.Es[s_key] != 0:
            return -self.Es[s_key]  # 結束局面：返回輸贏

        if s_key not in self.Ps:
            board_tensor = torch.tensor(state.get_canonical_board(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            board_tensor = board_tensor.to(self.args['device'])
            with torch.no_grad():
                log_pi, v = self.nnet(board_tensor)
            pi = torch.exp(log_pi).squeeze().cpu().numpy()
            legal_moves = state.get_legal_moves()
            mask = np.zeros(self.args['board_size']**2)
            for move in legal_moves:
                idx = move[0] * self.args['board_size'] + move[1]
                mask[idx] = 1
            pi *= mask
            pi /= np.sum(pi)

            self.Ps[s_key] = pi
            self.Vs[s_key] = mask
            self.Ns[s_key] = 0
            return -v.item()

        max_ucb = -float('inf')
        best_action = None
        board_size = self.args['board_size']
        for move in state.get_legal_moves():
            a = move[0] * board_size + move[1]
            if (s_key, a) in self.Qsa:
                u = self.Qsa[(s_key, a)] + self.args['cpuct'] * self.Ps[s_key][a] * math.sqrt(self.Ns[s_key]) / (1 + self.Nsa[(s_key, a)])
            else:
                u = self.args['cpuct'] * self.Ps[s_key][a] * math.sqrt(self.Ns[s_key] + 1e-8)
            if u > max_ucb:
                max_ucb = u
                best_action = a

        # print(f'best_action={best_action} board_size={board_size}')
        x, y = divmod(best_action, board_size)
        next_state = copy.deepcopy(state)
        next_state.make_move((x, y))
        v = self.search(next_state)

        if (s_key, best_action) in self.Qsa:
            self.Qsa[(s_key, best_action)] = (self.Nsa[(s_key, best_action)] * self.Qsa[(s_key, best_action)] + v) / (self.Nsa[(s_key, best_action)] + 1)
            self.Nsa[(s_key, best_action)] += 1
        else:
            self.Qsa[(s_key, best_action)] = v
            self.Nsa[(s_key, best_action)] = 1
        self.Ns[s_key] += 1
        return -v

    def get_action_probs(self, state, temp=1):
        for _ in range(self.args['num_mcts_sims']):
            self.search(copy.deepcopy(state))

        s_key = self._string_rep(state.board, state.current_player)
        counts = np.zeros(self.args['board_size']**2)
        for a in range(len(counts)):
            if (s_key, a) in self.Nsa:
                counts[a] = self.Nsa[(s_key, a)]

        if temp == 0:
            best_actions = np.argwhere(counts == np.max(counts)).flatten()
            probs = np.zeros_like(counts)
            probs[np.random.choice(best_actions)] = 1.0
            return probs
        else:
            counts = counts ** (1.0 / temp)
            return counts / np.sum(counts)

    def _string_rep(self, board, player):
        return str(board.reshape(-1)) + f"_{player}"
