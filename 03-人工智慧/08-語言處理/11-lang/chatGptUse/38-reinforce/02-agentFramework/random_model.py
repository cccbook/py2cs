import random

class Model:
  def __init__(self):
    # 不進行任何初始化
    pass
    
  def predict(self, state):
    # 隨機亂猜決策
    return random.randint(0, 1)
    
  def train(self, memory):
    # 不進行任何學習
    pass
