import bigPrime as bp

# 生成RSA金鑰對，金鑰長度為keySize
# 參考文獻 -- https://en.wikipedia.org/wiki/RSA_(cryptosystem)
def genKeyPair(keySize):
    # 生成兩個隨機質數p和q，長度為keySize位
    p = bp.randomPrime(keySize)
    q = bp.randomPrime(keySize)
    
    # 計算模數N = p * q，並計算r = (p-1) * (q-1)
    N = p * q
    r = (p - 1) * (q - 1)
    
    # 隨機選擇一個e，使得 1 < e < r 且 gcd(e, r) = 1，這裡選擇e為小於r的質數
    e = bp.randomPrime(keySize - 1)  # e的長度比r少1位
    
    # 計算d，使得 e * d ≡ 1 (mod r)，即d為e在模r下的逆元
    d = bp.modInv(e, r)
    
    # 返回公開金鑰(e, N)和私密金鑰(d, N)
    return e, d, N

# 使用公開金鑰(e, N)加密消息m
def encrypt(e, N, m):
    # 計算 m^e mod N，即對m進行指數模運算
    return bp.modPow(m, e, N)

# 使用私密金鑰(d, N)解密密文c
def decrypt(d, N, c):
    # 計算 c^d mod N，這樣可以得到原始消息m
    return bp.modPow(c, d, N)

if __name__ == '__main__':
    # 生成一對RSA金鑰對，金鑰長度為200位
    e, d, N = genKeyPair(200)
    
    # 輸出公開金鑰(e)和私密金鑰(d)以及模數(N)
    print('e=', e)
    print('d=', d)
    print('N=', N)
    
    # 生成一個隨機的大整數消息m，長度為100位
    m = bp.randomBigInt(100)
    
    # 使用公開金鑰加密消息m，得到密文c
    c = encrypt(e, N, m)
    
    # 使用私密金鑰解密密文c，得到解密後的消息m2
    m2 = decrypt(d, N, c)
    
    # 輸出原始消息m、加密後的密文c和解密後的消息m2
    print('m=', m)
    print('c=', c)
    print('m2=', m2)
    
    # 驗證解密後的消息m2是否與原始消息m相等
    assert m == m2  # 如果相等，則表示加密解密過程正常運作
