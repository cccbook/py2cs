"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    "蒙特卡羅樹搜索。先展開樹，然後選擇一個行動。"

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # 每個節點的累計獎勵 total reward of each node 
        self.N = defaultdict(int)  # 每個節點的累計訪問次數 total visit count for each node 
        self.children = dict()  # 每個節點的子節點 children of each node 
        self.exploration_weight = exploration_weight

    def choose(self, node): # 用 MCTS 搜尋樹找下法，主程式中的 board = tree.choose(board) 代表 AI 下子
        "Choose the best successor of node. (Choose a move in the game)" 
        "選擇節點的最佳後繼節點（選擇遊戲中的行動）"
        if node.is_terminal(): # 換 AI 下了，不可能已經結束
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children: # 如果 node 還沒展開，隨便找一個子 (child) 下
            return node.find_random_child()

        def score(n): # 取得分數 (平均獎勵)
            if self.N[n] == 0:
                return float("-inf")  # 避免未見過的行動 avoid unseen moves
            return self.Q[n] / self.N[n]  # 平均獎勵 average reward

        return max(self.children[node], key=score) # 選擇平均獎勵最高的那個下法 (child) 去下

    # 關鍵，這是整個 MCTS 的核心動作，包含 1. select 2. expand 3. simulate 4. backpropagate
    def do_rollout(self, node): # 一次模擬
        "Make the tree one layer better. (Train for one iteration.)"
        "將樹的層級增加一層（進行一次迭代訓練）"
        path = self._select(node) # 選擇下一條探索路徑 path
        leaf = path[-1] # 取得樹葉
        self._expand(leaf) # 展開該樹葉
        reward = self._simulate(leaf) # 模擬取得報酬 reward
        self._backpropagate(path, reward) # 獎勵 path 上的所有節點

    """
    _select 是在 MCTS 算法的搜索過程中，從給定的節點開始，遍歷 MCT 樹結構，
    尋找一個未探索的子孫節點，並返回從起始節點到該子孫節點的路徑。
    """
    def _select(self, node): 
        "Find an unexplored descendent of `node`"
        "尋找`node`的一個未探索的子孫節點"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:  # 該節點尚未探索 (或是終止節點) node is either unexplored or terminal 
                return path # 直接傳回該 path
            unexplored = self.children[node] - self.children.keys() # 否則，還有任何 child 沒探索嗎？
            if unexplored: # 如果有
                n = unexplored.pop() # 取得該未探索的 child
                path.append(n) # 加入 path
                return path # 傳回
            node = self._uct_select(node)  # 進入下一層 (選 UCT 最大的那個兒女) descend a layer deeper 

    def _expand(self, node): # 紀錄 node 的所有 children
        "Update the `children` dict with the children of `node`"
        "更新`children`字典以包含`node`的子節點"
        if node in self.children:
            return  # 已經展開 already expanded 
        self.children[node] = node.find_children() # 紀錄 node 的所有 children

    def _simulate(self, node): # 模擬對局直到遊戲結束，並取得 reward 加減分
        "Returns the reward for a random simulation (to completion) of `node`"
        "模擬對`node`進行隨機模擬（直到終局）的獎勵"
        invert_reward = True
        while True:
            if node.is_terminal(): # 若遊戲結束
                reward = node.reward() # 則取得回饋 reward
                return 1 - reward if invert_reward else reward # 根據角色決定加減分
            node = node.find_random_child()
            invert_reward = not invert_reward # 換人下了，所以分數評估必須反過來

    def _backpropagate(self, path, reward): # 用 reward 獎勵 path 上的所有節點
        "Send the reward back up to the ancestors of the leaf"
        "將獎勵傳播回葉節點的祖先節點"
        for node in reversed(path):
            self.N[node] += 1 # 增加累計訪問次數
            self.Q[node] += reward # 更新累計獎勵
            reward = 1 - reward  # 對於我來說是 1，對於敵人來說是 0，反之亦然 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node): # 用信賴區間上界 (Upper Confidence bound for Tree) 來選擇
        "Select a child of node, balancing exploration & exploitation"
        "選擇節點的一個子節點，平衡探索與利用"

        # 節點的所有子節點應該已經展開： All children of node should already be expanded: 
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n): # 計算信賴區間上界
            "Upper confidence bound for trees" 
            "用於樹的信賴區間上界"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct) # 選信賴區間上界最大的那個兒女


class Node(ABC): # 代表一個節點 (一盤棋)
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self): # 取得所有可能的下一子盤面
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self): # 隨機下子
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self): # 是否遊戲結束
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self): # 取得報酬 reward
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self): # 盤面雜湊 hash
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2): # 兩盤面是否相等
        "Nodes must be comparable"
        return True
