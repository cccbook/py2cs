import random

n = 1000000
inside = 0

for i in range(n):
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    if x**2 + y**2 < 1:
        inside += 1

pi = 4 * inside / n
print(pi)
