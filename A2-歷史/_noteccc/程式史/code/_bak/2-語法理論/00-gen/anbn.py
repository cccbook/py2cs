from random import *

def S():
    if random()<0.7:
        return f'a{S()}b'
    else:
        return ''

for _ in range(10):
    print(S())