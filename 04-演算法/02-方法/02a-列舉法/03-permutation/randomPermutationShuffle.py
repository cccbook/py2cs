# ChatGPT: https://chatgpt.com/share/66f22650-ebc8-8012-a9cb-8061012b51c5
import random

def random_permutations(elements, n):
    permutations = []
    for _ in range(n):
        # 複製元素，避免改變原始列表
        shuffled = elements[:]
        random.shuffle(shuffled)  # 隨機打亂元素
        permutations.append(shuffled)
    return permutations

# 測試範例
n = 10  # 產生 10 個排列
elements = list(range(1, 6))  # 產生 1 到 5 的元素
random_perms = random_permutations(elements, n)

# 列印結果
for i, p in enumerate(random_perms):
    print(f"排列 {i + 1}: {p}")
