from random_model import Model

class Agent:
  def __init__(self):
    # 初始化模型
    self.model = Model()
    self.memory = []
    
  def make_decision(self, state):
    # 根據狀態做出決策
    return self.model.predict(state)
    
  def record_results(self, state, action, next_state, reward, done):
    # 記錄行動和結果
    self.memory.append((state, action, next_state, reward, done))
    
  def update_learning(self):
    # 更新模型的學習
    self.model.train(self.memory)
