# 方法 d：用遞迴+查表
def power2n_d(n, memo={}):
    if n in memo:
        return memo[n]
    if n == 0:
        return 1
    result = power2n_d(n-1, memo) * 2
    memo[n] = result
    return result

# 測試函數
print('power2n_d(40)=', power2n_d(40))