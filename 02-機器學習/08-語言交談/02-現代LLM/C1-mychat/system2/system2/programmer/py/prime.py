def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def nth_prime(n):
    primes = []
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes[-1]

def test_nth_prime():
    # 測試前幾個質數是否正確
    assert nth_prime(1) == 2
    assert nth_prime(2) == 3
    assert nth_prime(3) == 5
    assert nth_prime(4) == 7
    assert nth_prime(5) == 11
    assert nth_prime(6) == 13
    
    # 測試第 10 個質數是否正確
    assert nth_prime(10) == 29
    
    print("所有測試通過！")

# 呼叫測試函數
test_nth_prime()
