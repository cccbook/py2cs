from qlearning import Q
import random

class QWalk1D(Q):
    def __init__(self):
        self.len = 6
        self.goal = 2
        self.q = {}
        for i in range(self.len+1):
            self.q[str(i)] = {'left':0, 'right':0}

    def isGoal(self, s):
        return s == self.goal

    def getStart(self):
        return random.randint(0, self.len)

    def chooseAction(self, s):
        a = random.choice(['left', 'right'])
        if s <= 0: return 'right'
        if s >= self.len: return 'left'
        return a

    def doAction(self, s, a):
        s1 = s+1 if a == 'right' else s-1
        r = 1 if self.isGoal(s1) else 0
        return (s1, r)

    def dump(self):
        q = self.q
        r = []
        for i in q:
            r.append(str(i) + ':l=' + str(q[i]['left']) + ' r=' + str(q[i]['right']))
        return '\n'.join(r)


q = QWalk1D()
q.learning(maxLoops=200, rate=0.1, decay=0.5)
