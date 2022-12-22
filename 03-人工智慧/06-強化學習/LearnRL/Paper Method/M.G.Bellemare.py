import random

class ClassGap:
    def __init__(self):
        self.eta = 3
        self.delta = 0.01
        self.gamma = 0.99
        self.value = [0, (-2/self.gamma)*(1+self.eta)]
        self.qvalue = [0, 0]
        self.reward = [0, 1]
        self.history = []
        self.action = random.randint(0,1)

    def eval(self):
        diff = 1
        while diff > self.delta:
            last_val = self.value
            if self.action == 1:
                self.value[0] = self.reward[0] + self.gamma*last_val[0]
            else:
                self.value[0] = self.reward[1] + self.gamma*0.5*sum(last_val) 
            for i in range(2):
                diff1 = abs(last_val[0] - self.value[0])
                diff2 = abs(last_val[1] - self.value[1])
            diff = max(diff1, diff2)
    
    def policy(self):
        last_policy = self.action
        self.qvalue[0] = 1 + self.gamma*0.5*sum(self.value)
        self.qvalue[1] = 0 + self.gamma*self.value[0]
        qvalue = self.qvalue
        print('Init action: ' + str(self.action))
        print('Q-value is ' + str(self.qvalue) + ', and action gap is ' + str(qvalue[1] - qvalue[0]))
        self.action = self.qvalue.index(max(qvalue))

class NovelGap:
    def __init__(self):
        self.eta = 3
        self.delta = 0.01
        self.gamma = 0.99
        self.value = [0, (-2/self.gamma)*(1+self.eta)]
        self.qvalue = [0, 0]
        self.reward = [0, 1]
        self.history = []
        self.action = random.randint(0,1)

    def eval(self):
        diff = 1
        while diff > self.delta:
            last_val = self.value
            if self.action == 1:
                self.value[0] = self.qvalue[1]
            else:
                self.value[0] = self.qvalue[0]
            for i in range(2):
                diff1 = abs(last_val[0] - self.value[0])
                diff2 = abs(last_val[1] - self.value[1])
            diff = max(diff1, diff2)

    def policy(self):
        last_policy = self.action
        last_qval = self.qvalue
        self.qvalue[0] = self.reward[1] + self.gamma*0.5*(self.value[1] + last_qval[0])
        self.qvalue[1] = self.reward[0] + self.gamma*last_qval[1]
        print('Init action: ' + str(self.action))
        print('Q-value is ' + str(self.qvalue) + ', and action gap is ' + str(self.qvalue[1] - self.qvalue[0]))
        self.action = self.qvalue.index(max(self.qvalue))

class MaxQGap:
    def __init__(self):
        self.eta = 3
        self.delta = 0.01
        self.gamma = 0.99
        self.value = [0, (-2/self.gamma)*(1+self.eta)]
        self.qvalue = [0, 0]
        self.reward = [0, 1]
        self.history = []
        self.action = random.randint(0,1)

    def eval(self):
        diff = 1
        while diff > self.delta:
            last_val = self.value
            if self.action == 1:
                self.value[0] = self.qvalue[1]
            else:
                self.value[0] = self.qvalue[0]
            for i in range(2):
                diff1 = abs(last_val[0] - self.value[0])
                diff2 = abs(last_val[1] - self.value[1])
            diff = max(diff1, diff2)

    def policy(self):
        last_policy = self.action
        last_qval = self.qvalue
        self.qvalue[0] = self.reward[1] + self.gamma*0.5*(self.value[1] + max(last_qval))
        self.qvalue[1] = self.reward[0] + self.gamma*max(last_qval)
        print('Init action: ' + str(self.action))
        print('Q-value is ' + str(self.qvalue) + ', and action gap is ' + str(self.qvalue[1] - self.qvalue[0]))
        self.action = self.qvalue.index(max(self.qvalue))

classtest = ClassGap()
for i in range(10):
    classtest.eval()
    classtest.policy()

newtest = NovelGap()
for i in range(10):
    newtest.eval()
    newtest.policy()

mytest = MaxQGap()
for i in range(10):
    mytest.eval()
    mytest.policy()