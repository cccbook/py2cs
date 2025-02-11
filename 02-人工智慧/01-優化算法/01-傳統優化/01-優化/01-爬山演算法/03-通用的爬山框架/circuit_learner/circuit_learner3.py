from truth_table import truth_tables
from hillClimbing import hillClimbing
import random

def init_tables():
    all_tables = []
    var_tables = [None]*4
    for i in range(0,4):
        tables = truth_tables(i)
        var_tables[i] = tables
        # print(tables)
        all_tables.append(tables)
    return tables, var_tables

all_tables, var_tables = init_tables()

def diff(circuit, target):
    score = 0
    oc = circuit['output']
    ot = target['output']
    for i in range(len(ot)):
        if oc[i]==ot[i]:
            score += 1
    return score

def adder4(a, b):
    # 確保輸入為 4 位元
    if len(a) != 4 or len(b) != 4:
        raise ValueError("Input arrays must have exactly 4 bits.")
    
    # 初始化變數
    result = [0] * 4  # 儲存結果
    carry = 0         # 進位

    # 從最低有效位（索引 0）到最高有效位（索引 3）逐位相加
    for i in range(4):
        sum_bit = (a[i] ^ b[i]) ^ carry   # 計算當前位元的和
        carry = (a[i] & b[i]) | ((a[i] ^ b[i]) & carry)  # 計算新的進位
        result[i] = sum_bit              # 儲存當前位元的結果

    # 返回結果和進位
    return result, carry

def height(s):

    ba1 = bitarray('1010')
    ba2 = bitarray('1100')
    return diff(s['sum'], sum_target) + diff(s['carry'], carry_target)

def neighbor(s):
    ns = {'sum':s['sum'], 'carry':s['carry']}
    choose = random.choice(['sum', 'carry'])
    ns[choose] = random.choice(var_tables[3])
    return ns

class Adder:
    def __init__(self, n):
        pass

    def run(a,b):
        return a+b

sum_target = {'inputs': [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], 'output': (0, 1, 1, 0, 1, 0, 0, 1)}
carry_target = {'inputs': [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], 'output': (0, 0, 0, 1, 0, 1, 1, 1)}
sum_circuit = var_tables[3][0]
carry_circuit = var_tables[3][0]
s = {'sum':sum_circuit,'carry':carry_circuit}

print('s=', s)
s = hillClimbing(s, height, neighbor)
print('s=', s)
