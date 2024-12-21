# ChatGPT -- https://chatgpt.com/share/66f216f7-16b0-8012-afe8-686b2d56e562
def generate_truth_table(n):
    def recursive_generate(n, current_row):
        # 當當前行達到指定長度時，將其打印
        if len(current_row) == n:
            print(current_row)
            return
        
        # 遞迴地生成 0 和 1 的組合
        recursive_generate(n, current_row + [0])
        recursive_generate(n, current_row + [1])

    # 從空列表開始遞迴
    recursive_generate(n, [])

# 設定變數數量 n
n = int(input("請輸入變數數量: "))
generate_truth_table(n)
