import math

s = 9999997

def seed(s):
    def f():
        global s
        s=(math.sin(s) * 10000)
        return s - math.floor(s)
    return f

random1 = seed(42)
random2 = seed(random1())
random = seed(random2())

for _ in range(10):
  print(random())
