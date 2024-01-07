import sys
from lexer import lex

tokens = None
ti = 0

# prog = stmts
def compile(code):
	global tokens, ti
	tokens = []
	for tk in lex(code):
		print(tk)
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
		print('tk = ', tk)
		ti += 1
		return tk.value
	else:
		error(f'在第 {tk.line} 行第 {tk.column} 個字有錯，應該是 {type} 但卻遇到 {tk.type}')

def next(value=None):
	global tokens, ti
	tk = tokens[ti]
	if value == None or isNext(value):
		print('tk = ', tk)
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
	print('STMT(): tk=', tokens[ti])
	if isNextT("BEGIN"):
		BLOCK()
	elif isNext("def"):
		FUNC()
	elif isNext("if"):
		IF()
	elif isNext("return"):
		RETURN()
	elif isNextT("ID"):
		id = next()
		if isNext("="):
			ASSIGN(id)
		else:
			CALL(id)
	else:
		error('不是一個陳述！')

# IF = if expr: stmt (elif stmt)* (else stmt)?
def IF():
	next('if')
	EXPR()
	next(':')
	STMT()
	while isNext('elif'):
		STMT()
	if isNext('else'):
		STMT()

# RETURN = return expr 
def RETURN():
	next('return')
	EXPR()

# ASSIGN: id = expr
def ASSIGN(id):
	next('=')
	EXPR()

# CALL: id(ARGS)
def CALL(id):
	next('(')
	ARGS()
	next(')')

# function = def id(params): block
def FUNC():
	next('def')
	nextT('ID')
	next('(')
	PARAMS()
	next(')')
	next(':')
	BLOCK()

# params = param*
def PARAMS():
	while not isNext(")"):
		PARAM()

# param = id
def PARAM():
	nextT("ID")

# block = : <begin> stmts <end>
def BLOCK():
	nextT('BEGIN')
	STMTS()
	nextT('END')

# expr = bexpr (if expr else expr)?
def EXPR():
	BEXPR()
	if isNext('if'):
		EXPR()
		next('else')
		EXPR()

# bexpr = mexpr (and|or) expr
def BEXPR():
	CEXPR()
	if isNextSet(['and', 'or']):
		op = next()
		EXPR()

# cexpr = mexpr ['==', '!=', '<=', '>=', '<', '>'] expr
def CEXPR():
	MEXPR()
	if isNextSet(['==', '!=', '<=', '>=', '<', '>']):
		op = next()
		EXPR()

# mexpr = item (['+', '-', '*', '/', '%'] expr)?
def MEXPR():
	ITEM()
	if isNextSet(['+', '-', '*', '/', '%']):
		op = next()
		EXPR()

# item = str | array | map | factor
# item = factor (now)
def ITEM():
	FACTOR()

# factor = (!-~) factor | num | ( expr ) | term
# factor = int | float | id | CALL | ( expr ) | term (now)
def FACTOR():
	if isNextT('FLOAT') or isNextT('INTEGER'):
		num = next()
	elif isNextT('ID'):
		id = next()
		if isNext('('):
			CALL(id)
	elif isNext('('):
		next('(')
		EXPR()
		next(')')
		TERM()

# term = id ( [expr] | . id | args )*
# term = id (args)?
def TERM():
	nextT('ID')
	if (isNext('(')):
		next('(')
		ARGS()
		next(')')

# array = [ expr* ]

# map = { (str:expr)* }

# args  = ( expr* ','? )
def ARGS():
	while not isNext(')'):
		EXPR()

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
