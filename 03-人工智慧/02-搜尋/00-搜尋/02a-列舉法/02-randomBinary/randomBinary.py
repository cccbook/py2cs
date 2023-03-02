import random

def randomBinaryString(n):
    r = []
    for i in range(n):
        r.append(random.choice(['0','1']))
    return ''.join(r)

for i in range(10):
    print(randomBinaryString(10))
