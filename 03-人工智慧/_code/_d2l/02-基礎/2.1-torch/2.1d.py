import torch

X = torch.tensor([1.0, 2, 4, 8])
A = X.numpy()
B = torch.from_numpy(A)
print('X=', X)
print('A=', A)
print('B=', B)
print('type(A)=', type(A))
print('type(B)=', type(B))
