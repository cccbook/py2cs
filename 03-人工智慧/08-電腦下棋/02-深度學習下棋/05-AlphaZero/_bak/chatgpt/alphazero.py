import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

board_size = 15
num_simulations = 100
# 1. 定義五子棋遊戲邏輯
class Game:
    def __init__(self):
        self.board = np.zeros((15, 15))
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((15, 15), dtype=int)
        self.current_player = 1
        self.state_history = []
        self.action_history = []

        self.update_state()

    def get_valid_moves(self):
        valid_moves = []
        for i in range(15):
            for j in range(15):
                if self.board[i][j] == 0:
                    valid_moves.append((i, j))
        return valid_moves

    def make_move(self, move):
        self.board[move[0]][move[1]] = self.current_player
        self.current_player = 3 - self.current_player  # 切換玩家

    def get_pieces(self, player):
        pieces = []
        for i in range(15):
            for j in range(15):
                if self.board[i][j] == player:
                    pieces.append((i, j))
        return pieces

    def is_terminal(self):
        # 檢查行
        for i in range(15):
            for j in range(11):
                if self.board[i][j] != 0:
                    if (
                        self.board[i][j] == self.board[i][j + 1] == self.board[i][j + 2] == self.board[i][j + 3] == self.board[i][j + 4]
                    ):
                        return True

        # 檢查列
        for i in range(11):
            for j in range(15):
                if self.board[i][j] != 0:
                    if (
                        self.board[i][j] == self.board[i + 1][j] == self.board[i + 2][j] == self.board[i + 3][j] == self.board[i + 4][j]
                    ):
                        return True

        # 檢查主對角線
        for i in range(11):
            for j in range(11):
                if self.board[i][j] != 0:
                    if (
                        self.board[i][j] == self.board[i + 1][j + 1] == self.board[i + 2][j + 2] == self.board[i + 3][j + 3] == self.board[i + 4][j + 4]
                    ):
                        return True

        # 檢查副對角線
        for i in range(4, 15):
            for j in range(11):
                if self.board[i][j] != 0:
                    if (
                        self.board[i][j] == self.board[i - 1][j + 1] == self.board[i - 2][j + 2] == self.board[i - 3][j + 3] == self.board[i - 4][j + 4]
                    ):
                        return True

        return False


    def get_winner(self):
        # 檢查行
        for i in range(15):
            for j in range(11):
                if (
                    self.board[i][j] == self.board[i][j + 1] == self.board[i][j + 2] == self.board[i][j + 3] == self.board[i][j + 4]
                    and self.board[i][j] != 0
                ):
                    return self.board[i][j]

        # 檢查列
        for i in range(11):
            for j in range(15):
                if (
                    self.board[i][j] == self.board[i + 1][j] == self.board[i + 2][j] == self.board[i + 3][j] == self.board[i + 4][j]
                    and self.board[i][j] != 0
                ):
                    return self.board[i][j]

        # 檢查主對角線
        for i in range(11):
            for j in range(11):
                if (
                    self.board[i][j] == self.board[i + 1][j + 1] == self.board[i + 2][j + 2] == self.board[i + 3][j + 3] == self.board[i + 4][j + 4]
                    and self.board[i][j] != 0
                ):
                    return self.board[i][j]

        # 檢查副對角線
        for i in range(4, 15):
            for j in range(11):
                if (
                    self.board[i][j] == self.board[i - 1][j + 1] == self.board[i - 2][j + 2] == self.board[i - 3][j + 3] == self.board[i - 4][j + 4]
                    and self.board[i][j] != 0
                ):
                    return self.board[i][j]

        return 0  # 若沒有贏家，返回 0 表示平局或遊戲還未結束


# 2. 定義神經網路模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 15 * 15, 128)
        self.fc2 = nn.Linear(128, 15 * 15 + 1)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(-1, 128 * 15 * 15)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_value = 0
        self.prior_probability = 0

    def select_child(self):
        best_child = None
        best_score = float("-inf")

        for child in self.children:
            score = child.total_value / child.visit_count + C * child.prior_probability * np.sqrt(2 * np.log(self.visit_count) / child.visit_count)

            if score > best_score:
                best_score = score
                best_child = child

        return best_child.state_action, best_child

    def expanded(self):
        return len(self.children) > 0

    def expand(self, action_probs):
        for action, prob in action_probs:
            new_state = self.state.make_move(action)
            new_child = Node(new_state, parent=self)
            new_child.prior_probability = prob
            self.children.append(new_child)

    def update(self, value):
        self.visit_count += 1
        self.total_value += value


    def get_best_action(self):
        best_visit_count = float("-inf")
        best_action = None

        for child in self.children:
            if child.visit_count > best_visit_count:
                best_visit_count = child.visit_count
                best_action = child.state_action

        return best_action

# 3. 定義AlphaZero代理
class Agent:
    def __init__(self, player):
        self.model = Net()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.player = player

    def train(self, num_epochs=100, num_games=1000):
        for epoch in range(num_epochs):
            for _ in range(num_games):
                # 進行自我對弈
                game = Game()
                self.play(game)

                # 獲取自我對弈的數據
                states, policies, values = game.get_data()

                # 將數據轉換為Tensor
                states = torch.tensor(states, dtype=torch.float)
                policies = torch.tensor(policies, dtype=torch.float)
                values = torch.tensor(values, dtype=torch.float)

                # 計算梯度
                self.optimizer.zero_grad()
                pred_policies, pred_values = self.model(states)
                loss = self.loss_fn(pred_policies, policies) + self.loss_fn(pred_values, values)
                loss.backward()

                # 更新模型參數
                self.optimizer.step()

            # 每個epoch打印訓練進度
            print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    def evaluate_state(self, state):
        # 这里使用简单的启发式评估函数来评估状态的价值
        # 可根据具体问题和需求设计更复杂的评估函数
        # 返回一个在 [-1, 1] 范围内的值，表示状态的估计价值

        if state.is_terminal():
            winner = state.get_winner()
            if winner == self.player:
                return 1.0  # 当前玩家获胜，返回最大价值
            elif winner == 3 - self.player:
                return -1.0  # 对手获胜，返回最小价值
            else:
                return 0.0  # 平局，返回中立价值

        # 如果状态不是终止状态，可以根据自定义的启发式规则计算状态的估计价值
        # 以下示例使用一个简单的计分函数：当前玩家的棋子数减去对手的棋子数
        player_pieces = state.get_pieces(self.player)
        opponent_pieces = state.get_pieces(3 - self.player)
        score = len(player_pieces) - len(opponent_pieces)

        # 归一化得分到 [-1, 1] 范围内
        normalized_score = score / (board_size ** 2)

        return normalized_score

    def play(self, game):
        root = Node(game.board)  # 創建根節點
        for _ in range(num_simulations):
            node = root
            # 選擇節點
            while node.expanded():
                action, node = node.select_child()
                game.make_move(action)

            # 擴展節點
            action_probs, value = self.evaluate_state(game)
            game_states.append(game.board.copy())
            policy_targets.append(action_probs)

            if not game.is_terminal():
                node.expand(action_probs)

            # 模擬
            value = self.simulate(game)
            game_states.append(game.board.copy())
            value_targets.append(value)

            # 回溯更新節點
            while node is not None:
                node.update(value)
                node = node.parent

            # 重置遊戲狀態
            game.reset()

        # 最佳走步的選擇
        best_action = root.get_best_action()
        game.make_move(best_action)


# 4. 訓練AlphaZero代理
agent = Agent(1)
agent.train()
