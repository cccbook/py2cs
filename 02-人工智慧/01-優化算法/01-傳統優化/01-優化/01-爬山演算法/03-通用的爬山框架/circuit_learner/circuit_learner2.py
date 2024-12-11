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

def height(s):
    return diff(s['sum'], sum_target) + diff(s['carry'], carry_target)

def neighbor(s):
    ns = {'sum':s['sum'], 'carry':s['carry']}
    choose = random.choice(['sum', 'carry'])
    ns[choose] = random.choice(var_tables[3])
    return ns

sum_target = {'inputs': [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], 'output': (0, 1, 1, 0, 1, 0, 0, 1)}
carry_target = {'inputs': [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], 'output': (0, 0, 0, 1, 0, 1, 1, 1)}
sum_circuit = var_tables[3][0]
carry_circuit = var_tables[3][0]
s = {'sum':sum_circuit,'carry':carry_circuit}

print('s=', s)
s = hillClimbing(s, height, neighbor)
print('s=', s)
