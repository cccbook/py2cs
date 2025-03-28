def gcd(x, y):
    while y != 0:
        x, y = y, x % y
    return x

print('gcd(27, 45)=', gcd(27, 45))
