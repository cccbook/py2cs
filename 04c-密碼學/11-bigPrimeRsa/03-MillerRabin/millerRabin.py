import random

# 計算 a^b % n 的快速冪
def power_mod(a, b, n):
    result = 1
    a = a % n
    while b > 0:
        if b % 2 == 1:
            result = (result * a) % n
        a = (a * a) % n
        b //= 2
    return result

# 米勒-拉賓質數判定法
def miller_rabin(n, k=5):
    # 特殊情況：1 不是質數，2 和 3 是質數
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    # 寫成 n-1 = 2^s * d 的形式
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # 進行 k 次隨機測試
    for _ in range(k):
        # 隨機選擇 a，1 <= a <= n-1
        a = random.randint(2, n - 2)
        # 計算 a^d % n
        x = power_mod(a, d, n)
        if x == 1 or x == n - 1:
            continue
        
        # 檢查是否存在 a^(2^r * d) % n == n-1
        for _ in range(s - 1):
            x = power_mod(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    # 如果經過多次測試都沒有找到錯誤，則 n 很可能是質數
    return True

# 測試
if __name__ == "__main__":
    test_numbers = [11, 15, 23, 33, 97, 100, 101]
    for num in test_numbers:
        result = miller_rabin(num)
        print(f"{num} is {'prime' if result else 'not prime'}")
