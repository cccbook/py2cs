from gate import *
from itertools import product
from hillClimbing import hillClimbing

def half_adder(a,b):
    s = gate_xor(a,b)
    c = gate_and(a,b)
    return s,c

def full_adder(a,b,c):
    s1,c1 = half_adder(a,b)
    s2,c2 = half_adder(s1,c)
    return s2, gate_or(c1,c2)

table = list(product([False, True], repeat=2))
print('ab|sc')
print('--|--')
for a,b in table:
    s,c = half_adder(a,b)
    print(f'{int(a)}{int(b)}|{int(s)}{int(c)}')

print()
table = list(product([False, True], repeat=3))
print('abc|sc')
print('---|--')
for a,b,c in table:
    s,c_out = full_adder(a,b,c)
    print(f'{int(a)}{int(b)}{int(c)}|{int(s)}{int(c_out)}')
