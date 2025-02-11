import torch

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print('a=', a)
print('b=', b)

print('a+b=', a+b)
print('a*b=', a*b)
