import itertools

def truth_tables(n):
    """
    生成包含輸入和所有可能輸出組合的真值表
    :param n: 輸入變數的數量
    :return: 包含輸入和輸出的所有組合
    """
    # 生成輸入的所有可能組合
    inputs = list(itertools.product([0, 1], repeat=n))
    
    # 生成輸出的所有可能組合
    outputs = list(itertools.product([0, 1], repeat=len(inputs)))
    
    # 組合輸入與輸出
    truth_table = []
    for output in outputs:
        truth_table.append({
            "inputs": inputs,
            "output": output
        })
    
    return truth_table

if __name__ == "__main__":
    # 範例：生成兩個輸入變數的真值表，包含 1-bit 輸出
    table2 = truth_tables(2)

    # 顯示結果
    for table in table2:
        print(f"Inputs: {table['inputs']}, Output: {table['output']}")
