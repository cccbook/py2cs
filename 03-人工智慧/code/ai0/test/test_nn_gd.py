from nn import Value, Vars

def test_nn_gd():
    x = Value(1.0)
    y = Value(2.0)
    loss = x*x+y*y
    model = Vars([x, y, loss])
    for k in range(100):
        loss = x*x+y*y
        model.zero_grad()
        loss.backward()
        # update (sgd)
        learning_rate = 1.0 - 0.9*k/100
        for p in model.parameters():
            p.data -= learning_rate * p.grad    
        if k % 1 == 0:
            print(f"step {k} x={x.data} y={y.data} loss={loss.data}")
    assert loss.data < 0.01
