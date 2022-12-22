import time
import math

SEED_MAX = 9999997
seed = time.time()%SEED_MAX

def random():
    global seed
    seed = (seed+37) % SEED_MAX
    x = math.sin(seed) * 93177
    return x - math.floor(x)

for _ in range(10):
  print(random())


