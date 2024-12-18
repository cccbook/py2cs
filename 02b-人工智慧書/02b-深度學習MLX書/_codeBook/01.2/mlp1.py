import mlx.core as mx
import mlx.nn as nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

x = mx.array([[1.0, 2.0]])
output = model(x)
print(output)
