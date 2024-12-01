import bigPrime as bp

# keySize=200

# reference -- https://en.wikipedia.org/wiki/RSA_(cryptosystem)
def genKeyPair(keySize):
    p = bp.randomPrime(keySize); q=bp.randomPrime(keySize)
    N = p*q; r=(p-1)*(q-1)
    e = bp.randomPrime(keySize-1) # e<r
    d = bp.modInv(e, r) # ed=1 mod r
    return e,d,N

def encrypt(e, N, m):
    return bp.modPow(m, e, N) # m^e mod N

def decrypt(d, N, c):
    return bp.modPow(c, d, N) # c^d = m^ed = m mod N

if __name__ == '__main__':
    e, d, N = genKeyPair(200)
    print('e=', e)
    print('d=', d)
    print('N=', N)
    m = bp.randomBigInt(100)
    c = encrypt(e, N, m)
    m2 = decrypt(d, N, c)
    print('m=', m)
    print('c=', c)
    print('m2=', m2)
    assert m==m2