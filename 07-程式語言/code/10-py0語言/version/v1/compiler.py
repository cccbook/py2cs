import sys
from lexer import lex

tokens = None
ti = 0

isDebugOn = False

def debug(*msg):
	if isDebugOn: print(*msg)

def emit(msg):
	print(msg, end='')

def indent():
	global level
	return '\t'*level

# prog = stmts
def compile(code):
	global tokens, ti
	tokens = []
	for tk in lex(code):
		debug(tk)
		tokens.append(tk)
	ti = 0
	STMTS()

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

def isEnd():
	global tokens, ti
	return ti >= len(tokens)

def error(msg):
	print(msg)
	raise Exception('compile error')
	sys.exit(1)

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

# stmts = stmt*
def STMTS():
	global ti
	while not isEnd() and not isNextT('END'):
		while isNextT('NEWLINE'): next()
		if isEnd(): break
		# print('ti=', ti)
		STMT()

"""
stmt = block                     |
	   function                  |
	   while expr: stmt           | 
	   if expr: stmt (elif stmt)* (else stmt)? |
	   return expr               |
	   assign
"""

def STMT():
	debug('STMT(): tk=', tokens[ti])
	emit(f'{indent()}')
	if isNextT("BEGIN"):
		BLOCK()
	elif isNext("def"):
		FUNC()
	elif isNext("if"):
		IF()
	elif isNext("return"):
		RETURN()
		emit(f';\n')
	elif isNextT("ID"):
		id = next(); emit(id)
		if isNext("="):
			ASSIGN(id)
		else:
			CALL(id)
		emit(f';\n')
	else:
		error('不是一個陳述！')

# IF = if expr: stmt (elif stmt)* (else stmt)?
def IF():
	next('if'); emit('if (')
	EXPR()
	next(':'); emit(')')
	STMT()
	while isNext('elif'):
		next('elif'); emit('else if')
		STMT()
	if isNext('else'):
		next('else'); emit('else')
		STMT()

# RETURN = return expr 
def RETURN():
	next('return'); emit('return ')
	EXPR()

# ASSIGN: id = expr
def ASSIGN(id):
	next('='); emit('=')
	EXPR()

# CALL: id(ARGS)
def CALL(id):
	next('('); emit('(')
	ARGS()
	next(')'); emit(')')

# function = def id(params): block
def FUNC():
	next('def')
	fname = nextT('ID')
	emit(f'int {fname}')
	next('('); emit('(')
	PARAMS()
	next(')'); emit(')')
	next(':')
	BLOCK()

# params = param*
def PARAMS():
	while not isNext(")"):
		PARAM()

# param = id
def PARAM():
	id = nextT("ID")
	emit(f'int {id}')

level = 0
# block = : <begin> stmts <end>
def BLOCK():
	global level
	nextT('BEGIN'); emit(f'\n{indent()}'+'{\n')
	level += 1
	STMTS()
	level -= 1
	nextT('END'); emit(f'\n{indent()}'+'}\n')

# expr = bexpr (if expr else expr)?
def EXPR():
	BEXPR()
	if isNext('if'):
		next('if')
		EXPR()
		next('else')
		EXPR()

# bexpr = cexpr ((and|or) cexpr)*
def BEXPR():
	CEXPR()
	while isNextSet(['and', 'or']):
		op = next(); cop = '&&' if op=='and' else 'or'; emit(f' {cop} ')
		CEXPR()

# cexpr = mexpr ['==', '!=', '<=', '>=', '<', '>'] mexpr
def CEXPR():
	MEXPR()
	while isNextSet(['==', '!=', '<=', '>=', '<', '>']):
		op = next(); emit(op)
		MEXPR()

# mexpr = item (['+', '-', '*', '/', '%'] item)?
def MEXPR():
	ITEM()
	while isNextSet(['+', '-', '*', '/', '%']):
		op = next(); emit(op)
		ITEM()

# item = str | array | map | factor
# item = factor (now)
def ITEM():
	FACTOR()

# factor = (!-~) factor | num | ( expr ) | term
# factor = int | float | id | CALL | ( expr ) | term (now)
def FACTOR():
	if isNextT('FLOAT') or isNextT('INTEGER'):
		num = next(); emit(num)
	elif isNextT('ID'):
		id = next(); emit(id)
		if isNext('('):
			CALL(id)
	elif isNext('('):
		next('('); next('(')
		EXPR()
		next(')'); next(')')
		TERM()

# term = id ( [expr] | . id | args )*
# term = id (args)?
def TERM():
	id = nextT('ID'); emit(id)
	if (isNext('(')):
		next('('); emit('(')
		ARGS()
		next(')'); emit(')')

# array = [ expr* ]

# map = { (str:expr)* }

# args  = ( expr* ','? )
def ARGS():
	while not isNext(')'):
		EXPR()
		if isNext(','):
			next(',')

# bool: True | False
# num : integer | float
# str : '...'
# id  : [a-zA-Z_][a-zA-Z_0-9]*

code = '''
def fib(n):
	if n == 0 or n == 1:
		a = 3
		return 1
	return fib(n-1)+fib(n-2)

print(fib(5))
'''

# 測試詞彙掃描器
if __name__ == "__main__":
	compile(code)
