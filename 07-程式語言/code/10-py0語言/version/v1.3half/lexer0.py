from typing import NamedTuple
import re

class Token(NamedTuple):
	type: str
	value: str
	line: int
	column: int
	level: int

def tokenize(code):
	keywords = {'def', 'if', 'while', 'for', 'return', 'and', 'or', 'not', 'yield', 'raise', 'continue', 'break'}
	token_specification = [
		('STRING',   r'(".*?")|(\'.*?\')'),        # String
		('FLOAT',    r'\d+\.\d*'),     # Float
		('INTEGER',  r'\d+'),          # Integer
		('ID',       r'[A-Za-z_]\w*'), # Identifiers
		('OP2',      r'(==)|(!=)|(<=)|(>=)'),    # Arithmetic operators
		('INDENT',   r'\n\t*'),        # Line indent
		('SPACE',    r'[ \t]+'),       # Skip over spaces and tabs
		('CHAR',     r'[{}()\+\-\*/=!:<>,&|^~]'), # 
		('MISMATCH', r'.'),            # Any other character
	]
	tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
	line_num = 1
	line_start = 0
	level = 0
	for mo in re.finditer(tok_regex, code):
		level_now = level
		kind = mo.lastgroup
		value = mo.group()
		column = mo.start() - line_start
		if kind == 'ID' and value in keywords:
			kind = value
		elif kind == 'INDENT':
			hlevel = len(value)-1
			if tokenize.lastTk and tokenize.lastTk.type in ['BEGIN', 'END', 'INDENT']:
				continue
			elif hlevel > level:
				kind = 'BEGIN'
				level_now = level
			elif hlevel < level: #  and tokenize.lastTk and not tokenize.lastTk.type in ['BEGIN', 'END', 'INDENT']
				kind = 'END'
				level_now = tokenize.lastTk.level - 1
			else:
				kind = 'NEWLINE' # 'INDENT'
				level_now = hlevel
				
			line_start = mo.start() + 1 # mo.end()
			line_num += 1
			level = hlevel
		elif kind == 'SPACE':
			continue
		elif kind == 'MISMATCH':
			raise RuntimeError(f'{value!r} unexpected on line {line_num}')
		tk = Token(kind, value, line_num, column, level_now)
		tokenize.lastTk = tk
		yield tk

tokenize.lastTk = None

def lex(code):
	tokens = []
	for tk in tokenize(code):
		tokens.append(tk)
	return tokens

def format(code):
	words = []
	for tk in tokenize(code):
		tabs = '\t'*tk.level
		if tk.type == 'BEGIN':
			words.append('\n'+tabs+'\t')# words.append('\n'+tabs+'begin\n'+tabs+'\t') # 多一個 \t ，因為 begin 後內縮一層
		elif tk.type == 'END':
			words.append('\n'+tabs+'\n'+tabs) # words.append('\n'+tabs+'end\n'+tabs)
		elif tk.type == 'NEWLINE':
			words.append('\n'+tabs)
		else:
			words.append(tk.value+' ')
	return ''.join(words)


# 測試詞彙掃描器
if __name__ == "__main__":
	from test0 import code
	tokens = lex(code)
	print(tokens)
	fcode = format(code)
	print(fcode)
