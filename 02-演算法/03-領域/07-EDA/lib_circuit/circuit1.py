# https://pypi.org/project/circuit/
from circuit import *

print('and gate:')
c = circuit()
g0 = c.gate(op.id_, is_input=True)
g1 = c.gate(op.id_, is_input=True)
g2 = c.gate(op.and_, [g0, g1])
g3 = c.gate(op.id_, [g2], is_output=True)
print('c.count()=', c.count()) # Number of gates in the circuit.
print('c.gate.to_legible()=', c.gate.to_legible())
print('c.evaluate([0, 1])=', c.evaluate([0, 1]))
print('truth table:', [list(c.evaluate(bs)) for bs in [[0, 0], [0, 1], [1, 0], [1, 1]]])
