import torch 
'''
x = net.variable(1)
y = net.variable(3)
x2 = net.mul(x, x)
y2 = net.mul(y, y)
o  = net.add(x2, y2)

print('net.forward()=', net.forward())
print('net.backwward()')
net.backward()
'''
x = torch.tensor(1., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
x2 = x*x
y2 = y*y
o = x2+y2

o.backward()

print(x.grad)    # x.grad = 2 
print(y.grad)    # y.grad = 6
print(o)    # o.value = 10
print(o.item())