# 定義初始假設 (最一般的假設)
def initial_hypothesis(num_features):
    return ['?' for _ in range(num_features)]  # 最初對每個特徵都不做任何限制

# 檢查假設是否與樣本匹配
def match_hypothesis(hypothesis, sample):
    return all(h == s or h == '?' for h, s in zip(hypothesis, sample))

# 更新最具體的假設 S
def update_specific_boundary(S, positive_sample):
    for i in range(len(S)):
        if S[i] == '?':
            S[i] = positive_sample[i]  # 設定為正樣本的具體特徵
        elif S[i] != positive_sample[i]:
            S[i] = '?'  # 將特徵泛化為 '?'
    return S

# 更新最一般的假設 G
def specialize_hypothesis(g, negative_sample):
    specializations = []
    for i in range(len(g)):
        if g[i] == '?':  # 只有 '?' 可以被具體化
            # 將該特徵具體化為負樣本的特徵
            new_g = g[:i] + [negative_sample[i]] + g[i+1:]
            specializations.append(new_g)
    return specializations

def update_general_boundary(G, negative_sample):
    G_new = []
    for g in G:
        if match_hypothesis(g, negative_sample):
            # 如果假設 g 匹配負樣本，則需具體化
            G_new.extend(specialize_hypothesis(g, negative_sample))
        else:
            G_new.append(g)  # 保留不匹配的假設
    return G_new

# 定義動物數據集
data = [
    (['是', '不下蛋', '四條腿'], 'positive'),  # A: 哺乳動物
    (['否', '下蛋', '四條腿'], 'negative'),    # B: 非哺乳動物
    (['是', '不下蛋', '非四條腿'], 'positive'), # C: 哺乳動物
    (['否', '下蛋', '非四條腿'], 'negative')   # D: 非哺乳動物
]

# 新增一個負樣本
new_negative_example = (['是', '下蛋', '四條腿'], 'negative')  # 這是一個負樣本

# 初始化 S 和 G
num_features = 3
S = initial_hypothesis(num_features)  # 最具體的假設
G = [initial_hypothesis(num_features)]  # 最一般的假設，初始為全 '?'

# 遍歷數據集更新 version space
for sample, label in data:
    if label == 'positive':  # 正樣本 (哺乳動物)
        S = update_specific_boundary(S, sample)  # 更新 S
        G = [g for g in G if match_hypothesis(g, sample)]  # 更新 G
    elif label == 'negative':  # 負樣本 (非哺乳動物)
        G = update_general_boundary(G, sample)  # 更新 G

# 對於新添加的負樣本進行處理
G = update_general_boundary(G, new_negative_example[0])

# 輸出最終的 S 和 G
print("最具體的假設 S:", S)
print("最一般的假設 G:", G)
