import random

def S():
    # random.choice([f'a{S()}b'], '')
    return '' if random.random()<0.5 else f'a{S()}b' 

for _ in range(10):
    print(S())