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

target = {'inputs': [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)], 'output': (0, 1, 1, 0, 1, 0, 0, 1)}

def height(circuit):
    score = 0
    oc = circuit['output']
    ot = target['output']
    for i in range(len(ot)):
        if oc[i]==ot[i]:
            score += 1
    return score

def neighbor(circuit):
    return random.choice(var_tables[3])

circuit = var_tables[3][0]
print('circuit=', circuit)
print('target=', target)
circuit = hillClimbing(circuit, height, neighbor)
print('circuit=', circuit)
print('target=', target)
