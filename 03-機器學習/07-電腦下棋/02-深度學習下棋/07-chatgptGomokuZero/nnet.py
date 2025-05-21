# nnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GomokuNNet(nn.Module):
    def __init__(self, board_size=9, n_channels=64):
        super(GomokuNNet, self).__init__()
        self.board_size = board_size

        # 共用的卷積層
        self.conv1 = nn.Conv2d(1, n_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(n_channels, n_channels, 3, padding=1)

        # 策略頭
        self.policy_conv = nn.Conv2d(n_channels, 2, 1)  # 1x1 卷積
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # 價值頭
        self.value_conv = nn.Conv2d(n_channels, 1, 1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, 1, board_size, board_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)  # log 機率分佈，便於與 MCTS 整合

        # value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # 範圍 [-1, 1]

        return p, v
