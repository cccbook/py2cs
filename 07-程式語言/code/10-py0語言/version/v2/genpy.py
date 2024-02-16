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
		case 'for':
			emit('for ')
			emit(n['id'])
			emit(' in ')
			gen(n['expr'])
			emit(' : ')
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
		case 'array': # ARRAY  = [ (EXPR ,)* EXPR? ]
			emit('[')
			elist = n['list']
			if len(elist)>0:
				for expr in elist[0:-1]:
					gen(expr)
					emit(',')
				gen(elist[-1])
			emit(']')
		case 'map':
			emit('{')
			pairs = n['pairs']
			for pair in pairs[0:-1]:
				emit(pair['key']+':')
				gen(pair['value'])
				emit(',')
			emit(pairs[-1]['key']+':')
			gen(pairs[-1]['value'])
			emit('}')
		case 'obj': # OBJ = id | string | int | float | LREXPR
			if n['type'] == 'lrexpr':
				gen(n['expr'])
			else:
				emit(n['value'])
		case 'term': # TERM   = OBJ ( [EXPR] | . id | (ARGS) )*
			tlist = n['list']
			obj = tlist[0]
			gen(obj['obj'])
			for t in tlist[1:]:
				op = t['type']
				if op == 'index':
					emit('[')
					gen(t['expr'])
					emit(']')
				elif op == 'member':
					emit('.'+t['id'])
				elif op == 'call':
					emit('(')
					gen(t['args'])
					emit(')')
				else:
					error(f'term: op = {op} 不合法！')
		case 'args': # ARGS = (EXPR ',')* EXPR? # args
			args = n['args']
			if len(args)>0:
				for arg in args[0:-1]:
					gen(arg)
					emit(' , ')
				gen(args[-1])
		case 'float'|'integer':
			emit(n['value'])
		case 'string':
			emit(n['value'])
		case 'id':
			emit(n['id'])

# 測試詞彙掃描器
if __name__ == "__main__":
	from test0 import code
	ast = parse(code)
	print(ast)
	gen(ast)
