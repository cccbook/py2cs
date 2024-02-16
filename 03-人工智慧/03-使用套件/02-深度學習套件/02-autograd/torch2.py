import torch 

x = torch.tensor(1.)
y = torch.tensor(3.)
x2 = x*x
y2 = y*y
o = x2+y2

'''
o.backward()

print(x.grad)    # x.grad = 2 
print(y.grad)    # y.grad = 6
'''
learning_rate = 1e-3
for t in range(100):
    loss = o.item()
    print(t, 'loss=', loss, 'o=', o.item())
    o.backward()
    with torch.no_grad():
        x -= learning_rate * x.grad
        y -= learning_rate * y.grad
        # x.grad.zero_()
        # y.grad.zero_()
