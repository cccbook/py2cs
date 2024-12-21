# Euclid's Algorithm 
# https://stackoverflow.com/questions/32042240/euclids-algorithm-javascript

def gcd(x, y):
    while y != 0:
        x, y = y, x % y
    return x

print(gcd(462, 910))