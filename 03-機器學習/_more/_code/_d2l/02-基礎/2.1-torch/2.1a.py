import torch

x = torch.arange(12, dtype=torch.float32)
print('x=', x)
print('x.numel()=', x.numel())
print('x.shape=', x.shape)
X = x.reshape(3, 4)
print('X=', X)
print('X.shape=', X.shape)

print('X[-1]=', X[-1])
print('X[1:3]=', X[1:3])

X[1, 2] = 17
print('X[1, 2] = 17')
print('X=', X)

X[:2, :] = 12
print('X[:2, :] = 12')
print('X=', X)

print('x=', x)
print('torch.exp(x)=', torch.exp(x))

