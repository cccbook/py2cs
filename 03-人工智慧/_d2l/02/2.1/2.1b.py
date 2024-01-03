import torch

print('torch.zeros((2, 3, 4))=', torch.zeros((2, 3, 4)))
print('torch.ones((2, 3, 4))=', torch.ones((2, 3, 4)))
print('torch.randn(3, 4)=', torch.randn(3, 4))
print('torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])=', 
      torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

x = torch.tensor([1, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print('x=', x)
print('y=', y)
print('x+y=', x+y)
print('x-y=', x-y)
print('x*y=', x*y)
print('x/y=', x/y)
print('x**y=', x**y)
print('x==y', x==y)
print('x<=y', x<=y)
print('x.dot(y)', x.dot(y))
print('x.sum(y)', x.sum())

