def fibonacci(n):
    if n == 0: return 0
    elif n == 1 or n == 2:
        return 1 
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def test_fibonacci():
    # 測試費氏數列的前幾個數字是否正確
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(2) == 1
    assert fibonacci(3) == 2
    assert fibonacci(4) == 3
    assert fibonacci(5) == 5
    assert fibonacci(6) == 8
    assert fibonacci(7) == 13
    assert fibonacci(8) == 21
    assert fibonacci(9) == 34
    assert fibonacci(10) == 55
    
    # 測試費氏數列的第 30 個數字是否正確
    assert fibonacci(30) == 832040
    
    print("所有測試通過！")

# 呼叫測試函數
test_fibonacci()