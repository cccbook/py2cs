import random

def randomBigInt(len):
    return random.randrange(10**len)

def modPow(b, e, n):
    b = b % n
    r = 1
    while (e > 0):
        if e % 2 == 1:
            r = (r * b) % n
        e = e // 2
        b = (b ** 2) % n
    return r

# 參考 -- https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
# gcd(a,b) = ri = a*si+b*ti
# gcd(e,N) =  1 = e*si+N*ti
#             1 = e*si mod N  => si=d 是 e 的 modInverse
def extEuclid(a, b):
    si, s = 1, 0
    ti, t = 0, 1
    ri, r = a, b
    if b == 0:
        return [1, 0, a]
    else:
        while (r != 0):
            q = ri // r
            ri, r = r, ri-q*r
            si, s = s, si-q*s
            ti, t = t, ti-q*t
    return [si, ti, ri]

# https://ithelp.ithome.com.tw/articles/10236425
# 擴展歐幾里得算法的等效公式：ax+by = gcd(a,b)。
# gcd(a,b)=ri=1=a*si+b*ti
# gcd(x,N)=1=x*si+N*ti   => x*si=1 mod N
def modInv(x, N):
    si,_,_ = extEuclid(x, N)
    return (si+N)%N

# ===================== millerRabinPrime() test======================
# Fermat 定理：若 n 是質數，則 a^{n-1} mod n = 1
# Pseudo Prime 偽質數：若 a^{n-1} mod n = 1
def decompose(m): # m=2^t * u
    u = m
    t = 0
    while (u%2 == 0):
        u = u // 2
        t += 1
    return t, u

def witness(a, n):
    t, u = decompose(n - 1)
    x = modPow(a, u, n)
    for i in range(1, t+1): # (let i = 1n; i <= t; i++) {
        xn = modPow(x, 2, n)
        if xn == 1 and x != 1 and x != n - 1:
            return True
        x = xn
    if x != 1: return True
    return False

def millerRabinPrime(n, s):
    for i in range(1, s+1):
        a = random.randrange(0, n)
        if witness(a, n):
          return False
    return True

def isPrime(n):
    return millerRabinPrime(n, 10)

def randomBigMayPrime(len):
    return randomBigInt(len-1)*10+random.choice([1,3,7,9])

# 找不到就傳回 null
def randomPrime(len, maxLoops = 9999999):
    r = None; failCount=0
    for i in range(0, maxLoops):
        r = randomBigMayPrime(len)
        if isPrime(r):
            break
        else:
            failCount += 1
    return r

if __name__ == '__main__':
    print('randomBigInt(100)=', randomBigInt(100))
    print('randomPrime(5)=', randomPrime(5))
    print('randomPrime(10)=', randomPrime(10))
    print('randomPrime(100)=', randomPrime(100))
    print('randomPrime(200)=', randomPrime(200))