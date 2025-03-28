
def adder4(a, b):
    # 確保輸入為 4 位元
    if len(a) != 4 or len(b) != 4:
        raise ValueError("Input arrays must have exactly 4 bits.")
    
    # 初始化變數
    result = [0] * 4  # 儲存結果
    carry = 0         # 進位

    # 從最低有效位（索引 0）到最高有效位（索引 3）逐位相加
    for i in range(3,-1,-1):
        sum_bit = (a[i] ^ b[i]) ^ carry   # 計算當前位元的和
        carry = (a[i] & b[i]) | ((a[i] ^ b[i]) & carry)  # 計算新的進位
        result[i] = sum_bit              # 儲存當前位元的結果

    # 返回結果和進位
    return result, carry

if __name__ == "__main__":
    c = adder4([1,0,0,1], [0,0,1,1])
    print(c)
