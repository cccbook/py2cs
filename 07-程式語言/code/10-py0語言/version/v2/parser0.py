from lib0 import *
from lexer0 import lex

tokens = None
ti = 0

def back():
	global ti
	ti -= 1

# prog = stmts
def parse(code):
	global tokens, ti
	tokens = lex(code)
	# print('tokens=', tokens)
	ti = 0
	return STMTS()

def isNextT(type):
	global tokens, ti
	if isEnd(): return False
	return tokens[ti].type == type
	
def isNext(value):
	global tokens, ti
	if isEnd(): return False
	return tokens[ti].value == value

def isNextSet(vset):
	global tokens, ti
	if isEnd(): return False
	return tokens[ti].value in vset

def isNextTSet(tset):
	global tokens, ti
	if isEnd(): return False
	return tokens[ti].type in tset

def isEnd():
	global tokens, ti
	return ti >= len(tokens)

def getTk():
	return tokens[ti]

def nextT(type):
	global tokens, ti
	tk = tokens[ti]
	if isNextT(type):
		debug('tk = ', tk)
		ti += 1
		return tk.value
	else:
		error(f'在第 {tk.line} 行第 {tk.column} 個字有錯，應該是 {type} 但卻遇到 {tk.type}')

def next(value=None):
	global tokens, ti
	tk = tokens[ti]
	if value == None or isNext(value):
		debug('tk = ', tk)
		ti += 1
		return tk.value
	else:
		error(f'在第 {tk.line} 行第 {tk.column} 個字有錯，應該是 {value} 但卻遇到 {tk.value}')

# STMTS = STMT*
def STMTS():
	global ti
	stmts = []
	while not isEnd() and not isNextT('END'):
		while isNextT('NEWLINE'): next()
		if isEnd(): break
		s = STMT()
		stmts.append(s)
	return {'type':'stmts', 'stmts':stmts}

# STMT = BLOCK | FUNC | IF | WHILE | RETURN | ASSIGN | CALL

def STMT():
	s = None
	print('STMT(): tk=', tokens[ti])
	if isNextT("BEGIN"):
		s = BLOCK()
	elif isNext("def"):
		s = FUNC()
	elif isNext("if"):
		s = IF()
	elif isNext("while"):
		s = WHILE()
	elif isNext("for"):
		s = FOR()
	elif isNext("return"):
		s = RETURN()
	elif isNextT("ID"):
		id = next()
		if isNext("="):
			s = ASSIGN(id)
		else:
			back()
			s = TERM()
	else:
		error('不是一個陳述！')
	return {'type':'stmt', 'stmt':s}

# IF = if expr: stmt (elif stmt)* (else stmt)?
def IF():
	next('if')
	expr = EXPR()
	next(':')
	stmt = STMT()
	
	elifList = []
	while isNext('elif'):
		next('elif')
		e = EXPR()
		next(':')
		s = STMT()
		elifList.extend({'type':'elif', 'expr':e, 'stmt':s})
		
	elseStmt = None
	if isNext('else'):
		next('else')
		elseStmt = STMT()
	return {'type':'if', 'expr':expr, 'stmt':stmt, 'elifList':elifList, 'elseStmt':elseStmt}

# WHILE = while expr: stmt
def WHILE():
	next('while')
	e = EXPR()
	next(':')
	s = STMT()
	return {'type':'while', 'expr':e, 'stmt':s}

# FOR = for id in EXPR: STMT
def FOR():
	next('for')
	id = nextT('ID')
	next('in')
	e = EXPR()
	next(':')
	s = STMT()
	return {'type':'for', 'id':id, 'expr': e, 'stmt':s }

# RETURN = return expr 
def RETURN():
	next('return')
	e = EXPR()
	return {'type':'return', 'expr':e}

# ASSIGN = id = expr
def ASSIGN(id):
	next('=')
	e = EXPR()
	return {'type':'assign', 'id':id, 'expr':e}

"""
# CALL = f(ARGS)
def CALL(f):
	next('(')
	args = ARGS()
	next(')')
	return {'type':'call', 'f':f, 'args':args}
"""

# FUNC = def id(PARAMS): BLOCK
def FUNC():
	next('def')
	id = nextT('ID')
	next('(')
	params = PARAMS()
	next(')')
	next(':')
	block = BLOCK()
	return {'type':'func', 'id':id, 'params':params, 'block':block}

# PARAMS = PARAM*
def PARAMS():
	params = []
	while not isNext(")"):
		p = PARAM()
		params.append(p)
	return {'type':'params', 'params':params}

# PARAM = id
def PARAM():
	id = nextT("ID")
	return {'type':'param', 'id':id}

level = 0
# BLOCK = begin STMTS end
def BLOCK():
	nextT('BEGIN')
	s = STMTS()
	nextT('END')
	return {'type':'block', 'stmts':s }

# EXPR = BEXPR (if EXPR else EXPR)?
def EXPR():
	bexpr = BEXPR()
	if isNext('if'):
		next('if')
		expr1 = EXPR()
		next('else')
		expr2 = EXPR()
		return {'type':'expr', 'bexpr':bexpr, 'expr1':expr1, 'expr2':expr2 }
	else:
		return bexpr

# BEXPR = CEXPR ((and|or) CEXPR)*
def BEXPR():
	e = CEXPR()
	elist = [e]
	while isNextSet(['and', 'or']):
		op = next()
		e = CEXPR()
		elist.extend([op, e])
	return e if len(elist)==1 else {'type':'bexpr', 'list':elist}

# CEXPR = MEXPR (['==', '!=', '<=', '>=', '<', '>'] MEXPR)*
def CEXPR():
	e = MEXPR()
	elist = [e]
	while isNextSet(['==', '!=', '<=', '>=', '<', '>']):
		op = next()
		e = MEXPR()
		elist.extend([op, e])
	return e if len(elist)==1 else {'type':'cexpr', 'list':elist}

# MEXPR = ITEM (['+', '-', '*', '/', '%'] ITEM)*
def MEXPR():
	e = ITEM()
	elist = [e]
	while isNextSet(['+', '-', '*', '/', '%']):
		op = next()
		e = ITEM()
		elist.extend([op,e])
	return e if len(elist)==1 else {'type':'mexpr', 'list':elist}

# ITEM = ARRAY | MAP | FACTOR
def ITEM():
	if isNext('['):
		e = ARRAY()
	elif isNext('{'):
		e = MAP()
	else:
		e = FACTOR()
	return e

# ARRAY  = [ (EXPR ,)* EXPR? ]
def ARRAY():
	next('[')
	elist = []
	while not isNext(']'):
		e = EXPR()
		elist.append(e)
		if not isNext(']'):
			next(',')
	next(']')
	return {'type':'array', 'list': elist}

# MAP = { (PAIR ,)* PAIR? }
def MAP():
	next('{')
	pairs = []
	while not isNext('}'):
		key = nextT('STRING')
		next(':')
		value = EXPR()
		pairs.append({'type':'pair', 'key':key, 'value': value})
		if not isNext('}'): next(',')
	next('}')
	return  {'type':'map', 'pairs':pairs}

# FACTOR = (!-~)* TERM
def FACTOR():
	e = TERM()
	return e

# TERM   = OBJ ( [EXPR] | . id | (ARGS) )*
def TERM():
	obj = OBJ()
	tlist = [obj]
	while isNextSet(['(', '.', '[']):
		if isNext('('):
			next('(')
			args = ARGS()
			next(')')
			tlist.append({'type':'call', 'args':args})
		elif isNext('['):
			next('[')
			e = EXPR()
			next(']')
			tlist.append({'type':'index', 'expr':e})
		elif isNext('.'): # member
			next('.')
			id = nextT('ID')
			tlist.append({'type':'member', 'id':id})
	return  {'type':'term', 'list':tlist}

# OBJ = id | string | integer | float | LREXPR
# LREXPR = ( expr )
def OBJ():
	obj = None
	tk = getTk()
	ty = tk.type.lower()
	if ty in ['string', 'integer', 'float']:
		next()
		obj = {'type':ty, 'value':tk.value}
	elif isNextT('ID'):
		id = next()
		obj = {'type':'id', 'id':id}
	elif isNext('('):
		next('(')
		e = EXPR()
		next(')')
		obj = {'type':'lrexpr', 'expr':e}
	else:
		error(f'OBJ:type={ty} 錯誤！')
	return {'type':'obj', 'obj':obj }

# array = [ expr* ]

# map = { (str:expr)* }

# ARGS  = (EXPR ',')* EXPR?
def ARGS():
	args = []
	while not isNext(')'):
		e = EXPR()
		args.append(e)
		if isNext(','):
			next(',')
	return {'type':'args', 'args':args}

# bool: True | False
# num : integer | float
# str : '...'
# id  : [a-zA-Z_][a-zA-Z_0-9]*

# 測試詞彙掃描器
if __name__ == "__main__":
	from test0 import code
	ast = parse(code)
	print(ast)
