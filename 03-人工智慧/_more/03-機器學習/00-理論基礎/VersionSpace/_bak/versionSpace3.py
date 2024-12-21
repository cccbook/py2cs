# 定義初始假設 (最一般的假設)
def initial_hypothesis(num_features):
    return ['?' for _ in range(num_features)]  # 最初對每個特徵都不做任何限制

# 檢查假設是否與樣本匹配
def match_hypothesis(hypothesis, sample):
    return all(h == s or h == '?' for h, s in zip(hypothesis, sample))

# 更新 version space 的泛化邊界 (S 假設)
def update_specific_boundary(S, positive_sample):
    for i in range(len(S)):
        if S[i] == '?':  # 只有當當前的 S 是 '?' 時才更新
            S[i] = positive_sample[i]
        elif S[i] != positive_sample[i]:  # 如果 S 與樣本不同，泛化為 '?'
            S[i] = '?'
    return S

# 針對負樣本，產生新的 G 假設
def specialize_hypothesis(g, negative_sample):
    specializations = []
    for i in range(len(g)):
        if g[i] == '?':  # 只能具體化 '?' 的特徵
            if negative_sample[i] != '?':  # 只具體化和負樣本不同的部分
                new_g = g[:i] + [negative_sample[i]] + g[i+1:]
                specializations.append(new_g)
    return specializations

# 更新 version space 的一般邊界 (G 假設)
def update_general_boundary(G, negative_sample):
    G_new = []
    for g in G:
        if match_hypothesis(g, negative_sample):
            # 如果 g 匹配負樣本，我們需要具體化這個假設
            G_new.extend(specialize_hypothesis(g, negative_sample))
        else:
            # 如果 g 不匹配負樣本，保留它
            G_new.append(g)
    return G_new

# 定義動物數據集
data = [
    # 樣本格式: (有毛髮, 下蛋, 四條腿), 哺乳動物?
    (['是', '不下蛋', '四條腿'], 'positive'),  # A: 哺乳動物
    (['否', '下蛋', '四條腿'], 'negative'),    # B: 非哺乳動物
    (['是', '不下蛋', '非四條腿'], 'positive'), # C: 哺乳動物
    (['否', '下蛋', '非四條腿'], 'negative')   # D: 非哺乳動物
]

# 初始化 S 和 G
num_features = 3
S = initial_hypothesis(num_features)  # 最具體的假設
G = [['?', '?', '?']]  # 最一般的假設，初始為全 '?'

# 遍歷數據集更新 version space
for sample, label in data:
    if label == 'positive':  # 正樣本 (哺乳動物)
        # 如果 S 不匹配正樣本，則進行泛化
        if not match_hypothesis(S, sample):
            S = update_specific_boundary(S, sample)
        # 從 G 中刪除不符合正樣本的假設
        G = [g for g in G if match_hypothesis(g, sample)]
    elif label == 'negative':  # 負樣本 (非哺乳動物)
        # 如果 G 的假設能解釋負樣本，則將它們具體化
        G = update_general_boundary(G, sample)

# 輸出最終的 S 和 G
print("最具體的假設 S:", S)
print("最一般的假設 G:", G)
