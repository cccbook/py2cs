import math

seed = 371
SEED_MAX = 9999997

def random():
    global seed
    seed = (seed + 37 ) % SEED_MAX
    x = math.sin(seed) * 93177
    return x - math.floor(x)

for i in range(100):
    print('random()=', random())
