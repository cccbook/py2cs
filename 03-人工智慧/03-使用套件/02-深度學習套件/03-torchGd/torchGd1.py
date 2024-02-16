import torch
import math

dtype = torch.float
x = torch.randn((), dtype=dtype, requires_grad=True) # torch.linspace(-math.pi, math.pi, 2000)

def loss_fn(x):
	loss = x*x-4*x+4
	return loss

def GD(x, loss_fn, loop_max = 10000, learning_rate = 1e-3):
	for t in range(loop_max):
		loss = loss_fn(x)
		if t % 100 == 99:
			print(t, 'x=', x.item(), 'loss=', loss.item())
		loss.backward()
		with torch.no_grad():
			x -= learning_rate * x.grad
			x.grad = None

GD(x, loss_fn, loop_max = 5000)

print(f'Result: x = {x.item()} loss={loss_fn(x)}')