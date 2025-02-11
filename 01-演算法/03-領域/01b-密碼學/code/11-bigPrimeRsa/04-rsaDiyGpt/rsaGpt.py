import random
from math import gcd

# 擴展歐幾里得算法
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y

# 生成 RSA 密鑰對
def generate_keypair(bits=1024):
    # 生成兩個大質數 p 和 q
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def generate_large_prime(bits):
        while True:
            p = random.getrandbits(bits)
            if is_prime(p):
                return p

    p = generate_large_prime(bits // 2)
    q = generate_large_prime(bits // 2)
    
    while p == q:  # p 和 q 不能相同
        q = generate_large_prime(bits // 2)
    
    # 計算 n 和 phi(n)
    n = p * q
    phi_n = (p - 1) * (q - 1)

    # 選擇公鑰 e
    e = random.randint(2, phi_n - 1)
    while gcd(e, phi_n) != 1:  # e 和 phi(n) 必須互質
        e = random.randint(2, phi_n - 1)
    
    # 計算私鑰 d
    _, d, _ = extended_gcd(e, phi_n)
    d = d % phi_n  # 保證 d 是正數

    return (e, n), (d, n)

# 加密
def encrypt(plaintext, pubkey):
    e, n = pubkey
    ciphertext = [pow(ord(char), e, n) for char in plaintext]
    return ciphertext

# 解密
def decrypt(ciphertext, privkey):
    d, n = privkey
    plaintext = ''.join([chr(pow(char, d, n)) for char in ciphertext])
    return plaintext

# 範例使用
if __name__ == "__main__":
    # 生成公鑰和私鑰
    pubkey, privkey = generate_keypair(100)
    
    # 顯示密鑰
    print("公鑰:", pubkey)
    print("私鑰:", privkey)
    
    # 原始訊息
    message = "Hello, RSA!"
    print("原始訊息:", message)
    
    # 加密訊息
    encrypted = encrypt(message, pubkey)
    print("加密後訊息:", encrypted)
    
    # 解密訊息
    decrypted = decrypt(encrypted, privkey)
    print("解密後訊息:", decrypted)
