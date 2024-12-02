import sys
from parser0 import parse

fname = sys.argv[1]
with open(fname) as f:
    code = f.read()

ast = parse(code)

from genpy import gen
gen(ast)

