def gd(p, loss, model, max_loops=100000, dump_period=1000, step=0.01):
    for i in range(max_loops):
        ploss = loss(p)
        model.zero_grad()
        ploss.backward()
        learning_rate = step # learning_rate = 1.0 - 0.9*i/100
        for n in model.parameters():
            n.data -= learning_rate * n.grad

        if i%dump_period == 0: 
            print('{:05d}:loss={:.3f}'.format(i, ploss.data))
        if ploss.grad < 0.00001: # 如果步伐已經很小了，就停止吧！
            break

    print('{:05d}:loss={:.3f}'.format(i, ploss.data))
    return ploss # 傳回最低點！