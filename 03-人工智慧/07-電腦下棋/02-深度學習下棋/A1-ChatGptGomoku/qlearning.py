import random

class QLearningAgent:
    def __init__(self, board_size=16, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.board_size = board_size
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def get_state_key(self, board):
        """ 将棋盘状态转换为哈希键 """
        return str(board.reshape(self.board_size * self.board_size))

    def choose_action(self, state, valid_actions):
        """ 根据 epsilon-greedy 策略选择动作 """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = [self.q_table.get((state, action), 0) for action in valid_actions]
            max_q_value = max(q_values)
            max_actions = [action for action, q_value in zip(valid_actions, q_values) if q_value == max_q_value]
            return random.choice(max_actions)

    def update_q_value(self, state, action, reward, next_state, next_valid_actions):
        """ 更新 Q 值 """
        current_q = self.q_table.get((state, action), 0)
        if next_valid_actions:
            next_q_values = [self.q_table.get((next_state, a), 0) for a in next_valid_actions]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0

        # 更新规则
        self.q_table[(state, action)] = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

    def get_valid_actions(self, board):
        """ 获取所有有效的动作 """
        return [(x, y) for x in range(self.board_size) for y in range(self.board_size) if board[x, y] == 0]

