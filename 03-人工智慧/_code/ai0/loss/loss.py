def mse_loss(p, predict, x, y):
    total = 0
    for i in range(len(x)):
        total += (y[i]-predict(p,x[i]))**2
    return total

def softmax_loss(self, p, x, y):
    pass