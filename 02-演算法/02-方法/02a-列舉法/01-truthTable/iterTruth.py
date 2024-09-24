# ChatGPT -- https://chatgpt.com/share/66f216f7-16b0-8012-afe8-686b2d56e562

import itertools

def generate_truth_table(n):
    # 使用 itertools.product 生成所有可能的布林值組合
    truth_table = list(itertools.product([0, 1], repeat=n))
    
    # 打印真值表
    for row in truth_table:
        print(row)

# 設定變數數量 n
n = int(input("請輸入變數數量: "))
generate_truth_table(n)
