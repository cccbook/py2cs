def improveLoop(p, improve, max_loops=100000000000, max_fails=10000):
    fail = 0
    for _ in range(max_loops):
        p1 = improve(p)
        if (p1 == None):
            fail += 1
            if (fail >= max_fails): break
        else:
            fail = 0
            p = p1
    return p

if __name__=="__main__":
    import random
    x = [3,5,1,2,4]

    def improve(a):
        i = random.randrange(0, len(a)-1)
        if (a[i] > a[i+1]):
            a[i], a[i+1] = a[i+1], a[i]
            return a

    print('init:x=', x)
    x = improveLoop(x, improve)
    print('best:x=', x)