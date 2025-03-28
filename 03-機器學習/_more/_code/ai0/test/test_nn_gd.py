import nn

def loss(p):
    x,y=p
    return x*x+y*y

def test_nn_gd():
    print('test_nn_gd()')
    x = nn.Value(1.0)
    y = nn.Value(2.0)
    p = [x,y]
    ploss = loss(p)
    model = nn.Vars([x,y,ploss])
    ploss = nn.gd(p, loss, model, max_loops=200, dump_period=10)
    assert ploss.data < 0.01
