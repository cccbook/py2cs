import math

# 定義計算以 2 為底的對數函數
def log2(x):
    return math.log(x, 2)

# 計算熵（Entropy）
# 熵的公式：H(p) = - Σ p(i) * log2(p(i))
# p 是一個概率分佈的列表，該函數會返回該分佈的熵
def entropy(p):
    r = 0  # 初始化結果變數
    for i in range(len(p)):  # 遍歷概率分佈 p 的每個元素
        r += p[i] * log2(1 / p[i])  # 計算每個事件的 p(i) * log2(1/p(i))
    return r  # 返回總的熵值

# 計算交叉熵（Cross Entropy）
# 交叉熵的公式：H(p, q) = - Σ p(i) * log2(q(i))
# p 是實際分佈，q 是估計分佈
def cross_entropy(p, q):
    r = 0  # 初始化結果變數
    for i in range(len(p)):  # 遍歷概率分佈 p 的每個元素
        r += p[i] * log2(1 / q[i])  # 計算每個事件的 p(i) * log2(1/q(i))
    return r  # 返回總的交叉熵值

# 計算 KL 散度（KL Divergence）
# KL 散度的公式：D_KL(p || q) = Σ p(i) * log2(p(i) / q(i))
# p 是實際分佈，q 是估計分佈，KL 散度用來衡量兩個分佈之間的差異
def kl_divergence(p, q):
    r = 0  # 初始化結果變數
    for i in range(len(p)):  # 遍歷概率分佈 p 的每個元素
        r += p[i] * log2(p[i] / q[i])  # 計算每個事件的 p(i) * log2(p(i)/q(i))
    return r  # 返回總的 KL 散度

# 定義三個不同的概率分佈 p、q 和 r
p = [1/4, 1/4, 1/4, 1/4]  # 等概率分佈
q = [1/8, 1/4, 1/4, 3/8]  # 不同的分佈
r = [1/100, 1/100, 1/100, 97/100]  # 大部分概率集中在最後一個事件

# 打印出各個分佈 p、q 和 r
print('p=', p)
print('q=', q)
print('r=', r)

# 計算並打印每個分佈的熵
print('entropy(p)=', entropy(p))  # 計算並輸出 p 的熵
print('entropy(q)=', entropy(q))  # 計算並輸出 q 的熵
print('entropy(r)=', entropy(r))  # 計算並輸出 r 的熵

# 計算並打印 p 相對於 p 的交叉熵，這應該等於熵
print('cross_entropy(p,p)=', cross_entropy(p, p))  # 交叉熵，p 對 p

# 計算並打印 p 相對於 q 的交叉熵
print('cross_entropy(p,q)=', cross_entropy(p, q))  # 交叉熵，p 對 q

# 計算並打印 p 相對於 r 的交叉熵
print('cross_entropy(p,r)=', cross_entropy(p, r))  # 交叉熵，p 對 r

# 重複計算 p 和 q 的交叉熵
print('cross_entropy(p,q)=', cross_entropy(p, q))  # 交叉熵，p 對 q

# 計算並打印 p 和 q 之間的 KL 散度
print('kl_divergence(p,q)=', kl_divergence(p, q))  # p 和 q 之間的 KL 散度

# 計算並打印 p 的熵
print('entropy(p)=', entropy(p))  # 再次輸出 p 的熵

# 計算並打印熵與 KL 散度的和（這應該等於 p 和 q 的交叉熵）
print('entropy(p)+kl_divergence(p,q)=', entropy(p) + kl_divergence(p, q))  # 熵加上 KL 散度應該等於交叉熵
