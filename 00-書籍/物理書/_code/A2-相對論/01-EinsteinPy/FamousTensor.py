# from -- https://docs.einsteinpy.org/en/latest/examples/Predefined%20Metrics%20in%20Symbolic%20Module.html

import sympy
from sympy import simplify
from einsteinpy.symbolic import RicciScalar
from einsteinpy.symbolic.predefined import Schwarzschild, DeSitter, AntiDeSitter, Minkowski, find

sympy.init_printing()  # for pretty printing
sch = Schwarzschild()
s = sch.tensor()
print('sch.tensor()=', s)

m = Minkowski(c=1).tensor()
print('Minkowski(c=1).tensor()=', Minkowski(c=1).tensor())

d = DeSitter().tensor()
print('DeSitter().tensor()=', DeSitter().tensor())

ad = AntiDeSitter().tensor()
print('AntiDeSitter().tensor()=', AntiDeSitter().tensor())
