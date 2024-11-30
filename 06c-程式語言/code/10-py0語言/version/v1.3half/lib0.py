isDebugOn = False

def debug(*msg):
	if isDebugOn: print(*msg)

def emit(msg):
	print(msg, end='')

