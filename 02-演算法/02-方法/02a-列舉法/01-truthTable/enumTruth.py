# ChatGPT -- https://chatgpt.com/share/66f216f7-16b0-8012-afe8-686b2d56e562

def generate_truth_table(n):
    # 初始化真值表
    truth_table = []
    
    # 用二進制的方式生成所有可能的組合
    for i in range(2 ** n):
        row = []
        # 根據二進制的位數來生成每個變數的值
        for j in range(n):
            # 使用右移和取餘數運算得到變數值
            row.append((i >> (n - 1 - j)) % 2)
        truth_table.append(row)
    
    # 打印真值表
    for row in truth_table:
        print(row)

# 設定變數數量 n
n = int(input("請輸入變數數量: "))
generate_truth_table(n)
