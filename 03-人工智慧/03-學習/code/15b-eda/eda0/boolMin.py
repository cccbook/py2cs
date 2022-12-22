import random
from improveLoop import *

class TruthTable:
    def __init__(self, table, nIn, names):
        self.nIn = nIn
        self.nOut = len(table[0])-nIn
        self.inputs = []
        self.outputs = []
        self.iNames = names[:nIn]
        self.oNames = names[nIn:]
        for row in table:
            self.inputs.append(row[:nIn])
            self.outputs.append(row[nIn:])

    def rows(self):
        return len(self.inputs)

class Sat:
    def __init__(self):
        self.map = {} # 滿足該真值表的字典查表
        self.array = [] # 滿足該真值表的所有列

    def load(self, table, oi):
        for ti in range(table.rows()):
            input = table.inputs[ti]
            out = table.outputs[ti][oi]
            if out!='0':
                self.array.append(input) 
                self.map[input]=out

def boolMinimize(table):
    rules = []
    for oi in range(table.nOut):
        r = boolMin1(table, oi)
        rules.append(r)
    return '\n'.join(rules)

def boolMin1(table, oi):
    sat = Sat()
    sat.load(table, oi)
    if len(sat.array)==0:
        return table.oNames[oi]+'=0'
    improveLoop(sat, improve) # keep improving
    rules = greedyCover(sat, table, oi)
    exp = table.oNames[oi]+'='+toDnf(rules, table.iNames)
    return exp

def greedyCover(sat, table, oi):
    sat.array.sort(reverse=True, key=lambda x:x.count('-'))
    rules = []
    for rule in sat.array:
        rules.append(rule)
        if cover(rules, table, oi):
            break
    return rules

def toDnf(rules, names):
    terms = []
    for rule in rules:
        term = toTerm(rule, names)
        terms.append(term)
    return '+'.join(terms)

def toTerm(rule, names):
    term = ""
    for i in range(len(rule)):
        if rule[i]=='1':
            term += names[i]
        elif rule[i]=='0':
            term += names[i]+"'"
        else:
            pass
    return "1" if term=="" else term
    
def improve(sat):
    s1 = random.choice(sat.array)
    i = random.randrange(0, len(s1))
    if s1[i] == '-': return False # xx-x 選到 - 不用改進
    side2 = '0' if s1[i]=='1' else '1' # 把該位反過來
    s2 = f"{s1[0:i]}{side2}{s1[i+1:]}"
    if sat.map.get(s2)=='1': # 若反過來後也已滿足，則開始嘗試將 xx-x 加入改進之 # s1 是 satArray 抽出來的，所以 satMap.get(s1) 必然為 '1' ，不用再檢查
        s = f"{s1[0:i]}-{s1[i+1:]}" # s1,s2 只有在第 i 位不同，有0有1，所以將該位合併為 -
        if sat.map.get(s)==None: # 如果 xx-x 不在 satMap 中，就加入
            sat.map[s] = '1' # 將 xx-x 加入 satMap 與 satArray 中
            sat.array.append(s)
            return True # 改進成功
    return False

def cover(rules, table, oi):
    for i in range(table.rows()):
        if table.outputs[i][oi]=='0': continue # f(x)=0, 不須涵蓋。
        # f(x)=1, 至少要有一條規則滿足，就是涵蓋
        satisfy = False
        for rule in rules:
            if match(table.inputs[i], rule):
                satisfy = True
                break
        if not satisfy:
            return False
    return True

def match(row, rule):
    for i in range(len(row)):
        if rule[i]!='-' and row[i]!=rule[i]:
            return False
    return True

if __name__ == '__main__':
    table = [
    #abco1A0CPfR
    "00001000001",
    "00101001111",
    "01001000110",
    "01111001011",
    "10001100100",
    "10111101000",
    "11011100001",
    "11111101101"
    ]

    table = TruthTable(table, 3, ["a", "b", "c", "o", "one", "A", "zero", "C", "parity", "f", "Removable"])
    rules = boolMinimize(table)
    print(rules)