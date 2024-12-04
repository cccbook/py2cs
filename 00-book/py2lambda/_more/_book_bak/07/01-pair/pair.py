# 定義布林值
TRUE = lambda x: lambda y: x
FALSE = lambda x: lambda y: y

# Lists
CONS = lambda x:lambda y:lambda f:f(x)(y)
CAR  = lambda p:p(TRUE)
CDR  = lambda p:p(FALSE)

pair = CONS(3)(5)
print(f'pair=CONS(3)(5)')
print(f'CAR(pair)={CAR(pair)}')
print(f'CDR(pair)={CDR(pair)}')

print('------')

two_pair = CONS(2)(pair)
print(f'two_pair=CONS(pair)(pair)')
print(f'CAR(CAR(two_pair)={CAR(CAR(two_pair))}')
print(f'CDR(CDR(two_pair)={CDR(CDR(two_pair))}')
