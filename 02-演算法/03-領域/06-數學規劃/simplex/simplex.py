# https://geekrodion.com/blog/operations/simplex
'''
# 單體法的一般解題步驟可歸納如下：

把線性規劃問題的約束方程組表達成典範型方程組，找出基本可行解作為初始基本可行解。
若基本可行解不存在，即約束條件有矛盾，則問題無解。
若基本可行解存在，從初始基可行解作為起點，根據最優性條件和可行性條件，引入非基變數取代某一基變數，找出目標函數值更優的另一基本可行解。
按步驟3進行疊代,直到對應檢定數滿足最優性條件（這時目標函數值不能再改善），即得到問題的最優解。
若疊代過程中發現問題的目標函數值無界，則終止疊代。

# 最佳化過程

任取一個非基變數{\displaystyle {{x}_{e}}}{{x}_{e}}，使得{\displaystyle {{c}_{e}}>0}{{c}_{e}}>0。
選取一個基變數{\displaystyle {{x}_{d}}}{{x}_{d}}，使得{\displaystyle {{A}_{d,e}}>0}{{A}_{d,e}}>0，且最小化{\displaystyle {{b}_{d}}/{{A}_{d,e}}\;}{{{b}_{d}}}/{{{A}_{d,e}}}\;
執行轉軸操作pivot(d, e)，並轉到第一步繼續算法。
'''
import math
import numpy as np

def simplex(c, A, b):
    tableau = to_tableau(c, A, b)

    while can_be_improved(tableau):
        pivot_position = get_pivot_position(tableau)
        tableau = pivot_step(tableau, pivot_position)

    return get_solution(tableau)

def to_tableau(c, A, b):
    xb = [eq + [x] for eq, x in zip(A, b)]
    z = c + [0]
    return xb + [z]

def can_be_improved(tableau):
    z = tableau[-1]
    return any(x > 0 for x in z[:-1])

def get_pivot_position(tableau):
    z = tableau[-1]
    column = next(i for i, x in enumerate(z[:-1]) if x > 0)
    
    restrictions = []
    for eq in tableau[:-1]:
        el = eq[column]
        restrictions.append(math.inf if el <= 0 else eq[-1] / el)

    row = restrictions.index(min(restrictions))
    return row, column    

def pivot_step(tableau, pivot_position):
    new_tableau = [[] for eq in tableau]
    
    i, j = pivot_position
    pivot_value = tableau[i][j]
    new_tableau[i] = np.array(tableau[i]) / pivot_value
    
    for eq_i, eq in enumerate(tableau):
        if eq_i != i:
            multiplier = np.array(new_tableau[i]) * tableau[eq_i][j]
            new_tableau[eq_i] = np.array(tableau[eq_i]) - multiplier
   
    return new_tableau
    
def is_basic(column):
    return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1

def get_solution(tableau):
    columns = np.array(tableau).T
    solutions = []
    for column in columns[:-1]:
        solution = 0
        if is_basic(column):
            one_index = column.tolist().index(1)
            solution = columns[-1][one_index]
        solutions.append(solution)
        
    return solutions

c = [1, 1, 0, 0, 0]
A = [
    [-1, 1, 1, 0, 0],
    [ 1, 0, 0, 1, 0],
    [ 0, 1, 0, 0, 1]
]
b = [2, 4, 4]

solution = simplex(c, A, b)
print('solution: ', solution)
