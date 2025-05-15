# gomoku.py
import numpy as np

class GomokuGame:
    def __init__(self, board_size=9, n_in_row=5):
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.last_move = None

    def get_legal_moves(self):
        return list(zip(*np.where(self.board == 0)))

    def make_move(self, move):
        x, y = move
        if self.board[x, y] != 0:
            raise ValueError("Invalid move!")
        self.board[x, y] = self.current_player
        self.last_move = move
        self.current_player *= -1

    def check_win(self):
        if self.last_move is None:
            return 0
        x, y = self.last_move
        player = self.board[x, y]
        directions = [(1,0), (0,1), (1,1), (1,-1)]

        for dx, dy in directions:
            count = 1
            for i in range(1, self.n_in_row):
                nx, ny = x + i*dx, y + i*dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                else:
                    break
            for i in range(1, self.n_in_row):
                nx, ny = x - i*dx, y - i*dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                else:
                    break
            if count >= self.n_in_row:
                return player
        return 0  # no win yet

    def is_draw(self):
        return np.all(self.board != 0) and self.check_win() == 0

    def get_canonical_board(self):
        return self.board * self.current_player  # current player's perspective
