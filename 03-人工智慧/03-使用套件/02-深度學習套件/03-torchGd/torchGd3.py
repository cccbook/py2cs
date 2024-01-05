import torch
x = torch.tensor([1.0,2.0], requires_grad=True)

step = 0.01
for i in range(500):
    z = x.norm()
    z.backward()
    # print('z=', z)
    # print('x.grad=', x.grad)
    with torch.no_grad():
        x -= step * x.grad
        x.grad.zero_()
    # print('x=', x)
print('x=', x)