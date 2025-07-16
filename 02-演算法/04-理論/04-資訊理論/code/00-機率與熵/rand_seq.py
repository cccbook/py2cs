import random

rlist = []
for i in range(100):
    r = random.choice(['A', 'B', 'C', 'D'])
    rlist.append(r)

print(''.join(rlist))

