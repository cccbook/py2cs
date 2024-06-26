import sys
from kb import KB

kb1 = KB()
with open(sys.argv[1], encoding='utf-8') as file:
    code = file.read().replace(r'\n', '')

kb1.load(code)
kb1.forwardChaining()
