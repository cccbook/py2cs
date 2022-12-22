def improveLoop(sol, improve, maxTry=100000000, maxFail=10000):
    fail = 0
    for t in range(maxTry):
        if improve(sol):
            fail = 0
        else:
            fail += 1
        if (fail > maxFail): break
    return sol
