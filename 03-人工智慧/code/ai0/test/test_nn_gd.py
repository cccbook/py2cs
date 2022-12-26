import nn
# from nn import Value, Vars, gd

def loss(p):
    x,y=p
    return x*x+y*y

def test_nn_gd():
    print('test_nn_sgd()')
    x = nn.Value(1.0)
    y = nn.Value(2.0)
    p = [x,y]
    ploss = loss(p)
    model = nn.Vars([x,y,ploss])
    ploss = nn.gd(p, loss, model, 200, 10, 0.01)
    assert ploss.data < 0.01

'''
    rloss = loss([x,y])
    model = Vars([x,y,rloss])
    for k in range(100):
        rloss = loss([x,y])
        model.zero_grad()
        rloss.backward()
        # update (sgd)
        learning_rate = 1.0 - 0.9*k/100
        for p in model.parameters():
            p.data -= learning_rate * p.grad    
        if k % 1 == 0:
            print(f"step {k} x={x.data} y={y.data} loss={rloss.data}")
    assert rloss.data < 0.01


def test_nn_gd():
    x = Value(1.0)
    y = Value(2.0)
    rloss = loss([x,y])
    model = Vars([x,y,rloss])
    for k in range(100):
        rloss = loss([x,y])
        model.zero_grad()
        rloss.backward()
        # update (sgd)
        learning_rate = 1.0 - 0.9*k/100
        for p in model.parameters():
            p.data -= learning_rate * p.grad    
        if k % 1 == 0:
            print(f"step {k} x={x.data} y={y.data} loss={rloss.data}")
    assert rloss.data < 0.01
'''