# 參考 https://en.wikipedia.org/wiki/Q-learning
# https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-general-rl/
import json

class Q:
    def __init__(self):
        pass

    def learning(self, maxLoops, rate, decay):
        q = self.q
        for loops in range(maxLoops):
            s0 = self.getStart()
            s0s = json.dumps(s0)
            while not self.isGoal(s0):
                a = self.chooseAction(s0)
                s1, r = self.doAction(s0, a)
                s1s = json.dumps(s1)
                print('s0=', s0, 'a=', a, 's1=', s1, 's0s=', s0s, 's1s=', s1s)
                q[s0s][a] = (1-rate) * q[s0s][a] + rate * (r + decay * self.argmax(q[s1s])) # self.argmax(q[s1]) 是下個狀態的最大報酬
                s0 = s1
            print('======= dump: {} =====\n{}'.format(loops, self.dump()))

    def isGoal(self, s): raise # 判斷是否已到達目標

    def getStart(self): raise # 取得起始點

    def chooseAction(self, s): raise # 選取在 s 狀態的動作 a

    def doAction(self, s, a): raise # 在狀態 s 做 a 動作時，會跑到 s1 狀態，並得到 reward r
    
    def argmax(self, qs): # 取的報酬最大的狀態
        max = float('-inf')
        for k in qs:
            if qs[k] > max: max = qs[k]
        return max
    
    def dump(self): raise # 傳回想印出的狀態字串
