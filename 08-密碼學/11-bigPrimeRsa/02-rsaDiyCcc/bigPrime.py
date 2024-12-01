import random

# 生成隨機大整數，長度為 len 位數
def randomBigInt(len):
    return random.randrange(10**len)

# 指數模運算 b^e mod n
def modPow(b, e, n):
    b = b % n  # b對n取餘數，防止b過大
    r = 1  # 初始化結果為 1
    while (e > 0):  # 當指數e大於0時，繼續運算
        if e % 2 == 1:  # 如果e是奇數，則乘上b並對n取餘數
            r = (r * b) % n
        e = e // 2  # 整數除法，e每次減半
        b = (b ** 2) % n  # b平方並對n取餘數
    return r

# 擴展歐幾里得算法，求出 gcd(a, b) 且滿足 ax + by = gcd(a, b) 的 x, y
# https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
# 返回 x 和 y 使得 e * x + N * y = gcd(e, N)
def extEuclid(a, b):
    si, s = 1, 0
    ti, t = 0, 1
    ri, r = a, b
    if b == 0:
        return [1, 0, a]  # 如果b是0，則gcd(a, b) = a，x=1, y=0
    else:
        while (r != 0):  # 迭代直到 r 為0
            q = ri // r  # 求商
            ri, r = r, ri - q * r  # 更新餘數
            si, s = s, si - q * s  # 更新x值
            ti, t = t, ti - q * t  # 更新y值
    return [si, ti, ri]  # 返回x, y, gcd

# 求x對N的模反元素，即x的逆元
# x * si ≡ 1 (mod N)
def modInv(x, N):
    si, _, _ = extEuclid(x, N)  # 使用擴展歐幾里得算法計算逆元
    return (si + N) % N  # 確保逆元是正數

# ===================== Miller-Rabin質數測試 =======================
# Fermat 定理：若 n 是質數，則 a^(n-1) mod n = 1
# 偽質數：若 a^(n-1) mod n = 1，但 n 不是質數
def decompose(m):  # 把 m 分解成 2^t * u
    u = m
    t = 0
    while (u % 2 == 0):  # 將u除以2直到u是奇數
        u = u // 2
        t += 1
    return t, u  # 返回t和u

# 實行米勒-拉賓質數測試
def witness(a, n):
    t, u = decompose(n - 1)  # 分解 n-1
    x = modPow(a, u, n)  # 計算 a^u mod n
    for i in range(1, t + 1):  # 重複t次
        xn = modPow(x, 2, n)  # x平方後模n
        if xn == 1 and x != 1 and x != n - 1:
            return True  # 如果中途x變為1但x不是1或n-1，則n為合數
        x = xn
    if x != 1: return True  # 如果x最終不是1，則n為合數
    return False

# 執行Miller-Rabin質數測試s次
def millerRabinPrime(n, s):
    for i in range(1, s + 1):  # 測試s次
        a = random.randrange(0, n)  # 隨機選擇a
        if witness(a, n):  # 如果a不是見證者，n是合數
            return False
    return True  # 如果通過了所有測試，n是質數

# 判斷n是否為質數，默認進行10次Miller-Rabin測試
def isPrime(n):
    return millerRabinPrime(n, 10)

# 生成隨機大數質數，長度為len位數
def randomBigMayPrime(len):
    return randomBigInt(len - 1) * 10 + random.choice([1, 3, 7, 9])  # 確保數字以1, 3, 7, 9結尾

# 生成長度為len位的隨機質數，最多進行maxLoops次嘗試
def randomPrime(len, maxLoops=9999999):
    r = None
    failCount = 0
    for i in range(0, maxLoops):  # 最多maxLoops次嘗試
        r = randomBigMayPrime(len)
        if isPrime(r):  # 如果是質數則返回
            break
        else:
            failCount += 1  # 否則繼續嘗試
    return r

if __name__ == '__main__':
    # 測試不同長度的隨機大數及隨機質數
    print('randomBigInt(100)=', randomBigInt(100))
    print('randomPrime(5)=', randomPrime(5))
    print('randomPrime(10)=', randomPrime(10))
    print('randomPrime(100)=', randomPrime(100))
    print('randomPrime(200)=', randomPrime(200))
