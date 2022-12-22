def hillClimbing(p, neighbor, height, max_loops=100000000000, max_fails=10000):
    fail = 0
    for _ in range(max_loops):
        p1 = neighbor(p)
        if height(p1) <= height(p):
            fail += 1
            if (fail >= max_fails): break
        else:
            fail = 0
            p = p1
    return p

if __name__=="__main__":
    import random

    def height(x):
        return -1*(x*x-2*x+1)

    def neighbor(x, dx=0.01):
        step = random.choice([-dx, dx])
        x += step
        return x

    x = 0.0
    print('init:x=', x)
    x = hillClimbing(x, neighbor, height)
    print('best:x=', x)