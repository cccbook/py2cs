isDebugOn = False

def debug(*msg):
	if isDebugOn: print(*msg)

def emit(msg):
	print(msg, end='')

def error(msg):
	print(msg)
	raise Exception('parse error')