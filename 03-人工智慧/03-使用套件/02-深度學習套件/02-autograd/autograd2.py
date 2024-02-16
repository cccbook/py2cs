import torch
x = torch.tensor([1.0,2.0], requires_grad=True)
z = x.norm()
z.backward()
print('z=', z)
print('x.grad=', x.grad)