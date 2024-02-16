from lib0 import *
from parser0 import parse
	
def indent():
	global level
	return '\t'*level

level = 0

def gen(n):
	global level
	t = n['type']
	match t:
		case 'stmts':
			for stmt in n['stmts']:
				gen(stmt)
		case 'stmt':
			if n['stmt']['type'] != 'block': # block 也是一種 stmt，但不需要換行，因為裡面的 stmt* 每個都會換
				emit(f'\n{indent()}')
			gen(n['stmt'])
		case 'while':
			emit('while ')
			gen(n['expr'])
			emit(':')
			gen(n['stmt'])
		case 'if':
			emit('if ')
			gen(n['expr'])
			emit(':')
			gen(n['stmt'])
			for el in n['elifList']:
				emit('elif ')
				gen(el['expr'])
				emit(':')
				gen(el['stmt'])
			if n['elseStmt']:
				emit('else:')
				gen(n['elseStmt'])
		case 'return':
			emit('return ')
			gen(n['expr'])
		case 'assign':
			emit(f'{n["id"]} = ')
			gen(n['expr'])
		case 'func':
			emit(f'def {n["id"]}(')
			gen(n['params'])
			emit('):')
			gen(n['block'])
		case 'params':
			params = n['params']
			for param in params[0:-1]:
				gen(param)
				emit(',')
			gen(params[-1])
		case 'param':
			emit(n['id'])
		case 'block': # BLOCK  = begin STMTS end
			level += 1
			gen(n['stmts'])
			level -= 1
			emit('\n')
		case 'expr': # EXPR = BEXPR (if EXPR else EXPR)?
			gen(n['bexpr'])
			# (if EXPR else EXPR)? 尚未處理
		case 'mexpr'|'cexpr'|'bexpr':
			for e in n['list']:
				if isinstance(e, str): # op
					emit(f' {e} ')
				else:
					gen(e)
		case 'lrexpr': # LREXPR = ( EXPR )
			emit('(')
			gen(n['expr'])
			emit(')')
		case 'call': # CALL = id(ARGS)
			emit(f'{n["id"]}(')
			gen(n['args'])
			emit(')')
		case 'args': # ARGS = (EXPR ',')* EXPR? # args
			for arg in n['args']:
				gen(arg)
		case 'float'|'integer':
			emit(n['value'])
		case 'id':
			emit(n['id'])

code = '''
def fib(n):
	if n == 0 or n == 1:
		a = 3
		return 1
	return fib(n-1)+fib(n-2)

def sum(n):
	s = 0
	i = 1
	while i<n:
		s = s+i
		i = i+1
	return s

print(fib(5))
print(sum(10))
'''

# 測試詞彙掃描器
if __name__ == "__main__":
	ast = parse(code)
	print(ast)
	gen(ast)
