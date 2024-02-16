import torch

A = torch.arange(6).reshape(3, 2)
print('A=', A)
print('A.T=', A.T)

# B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 3, 5]])
print('B=', B)
print('B == B.T?', B==B.T)