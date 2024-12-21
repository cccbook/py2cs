def gate_and(a,b):
    return a and b

def gate_or(a,b):
    return a or b

def gate_not(a):
    return not a

def gate_xor(a,b):
    return a != b

if __name__=="__main__":
    a, b = False, True
    print(f'and(a,b)={gate_and(a,b)} or(a,b)={gate_or(a,b)} not(a)={gate_not(a)} xor(a,b)={gate_xor(a,b)}')

