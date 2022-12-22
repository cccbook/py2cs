def group(op, p) :
    print('=============', op, '===============')
    for a in range(0, p):
        print()
        for b in range(0, p):
            ab = a*b if op=='*' else a+b
            ab = ab % p
            print(a, op, b, '=', ab, 'mod', p)

group('+', 7)
group('*', 7)
