import torch
import math

dtype = torch.float
x = torch.randn((), dtype=dtype, requires_grad=True) # torch.linspace(-math.pi, math.pi, 2000)

def loss_fn(parameters):
	x = parameters[0]
	loss = x*x-4*x+4
	return loss

def GD(parameters, loss_fn, loop_max = 10000, learning_rate = 1e-3):
	for t in range(loop_max):
		loss = loss_fn(parameters)
		if t % 100 == 99:
			print(t, 'parameters=', parameters, 'loss=', loss.item())
		loss.backward()
		with torch.no_grad():
			for x in parameters:
				x -= learning_rate * x.grad
				x.grad = None

params = [x]
GD(params, loss_fn, loop_max = 5000)

print(f'Result: parameters = {params} loss={loss_fn(params)}')