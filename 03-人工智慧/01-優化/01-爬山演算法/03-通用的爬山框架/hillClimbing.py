def hillClimbing(x, height, neighbor, max_fail=10000):
    fail = 0
    while True:
        nx = neighbor(x)
        if height(nx)>height(x):
            x = nx
            fail = 0
        else:
            fail += 1
            if fail > max_fail:
                return x

