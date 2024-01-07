import sys
from compiler import compile
with open(sys.argv[1]) as f:
    pyCode = f.read()

cppCode = compile(pyCode)
print(cppCode)