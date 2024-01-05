import torch
x = torch.ones(1, requires_grad=True)
y = torch.ones(1, requires_grad=True)
z = x*x + y * y
z.backward()     # automatically calculates the gradient
print(x.grad)    # ∂z/∂x = 2