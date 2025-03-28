import numpy as np

# 假設這是提供的 R 中的數學運算對應 Python 的定義
class R:
    @staticmethod
    def xplog(e, p):
        # 計算 log-likelihood，這裡需要實現具體的細節
        return np.log(np.array(p)) * np.array(e)

    @staticmethod
    def exp(val):
        return np.exp(val)

    @staticmethod
    def mul(a, b):
        return np.multiply(a, b)

    @staticmethod
    def add(a, b):
        return np.add(a, b)

    @staticmethod
    def sub(a, b):
        return np.subtract(a, b)

    @staticmethod
    def max(a):
        return np.max(a)

# EM算法
def EM():
    # 1st:  Coin B, {HTTTHHTHTH}, 5H,5T
    # 2nd:  Coin A, {HHHHTHHHHH}, 9H,1T
    # 3rd:  Coin A, {HTHHHHHTHH}, 8H,2T
    # 4th:  Coin B, {HTHTTTHHTT}, 4H,6T
    # 5th:  Coin A, {THHHTHHHTH}, 7H,3T
    # 所以，從 MLE 得出：pA(heads) = 0.80 和 pB(heads) = 0.45
    e = [[5, 5], [9, 1], [8, 2], [4, 6], [7, 3]]
    pA = np.array([0.6, 0.4])
    pB = np.array([0.5, 0.5])
    gen = 0
    delta = 9.9999

    for gen in range(1000):
        if delta <= 0.001:
            break
        print(f"pA={pA} pB={pB} delta={delta:.4f}")
        sumA = np.array([0.0, 0.0])
        sumB = np.array([0.0, 0.0])

        for i in range(len(e)):
            lA = R.xplog(e[i], pA)
            lB = R.ç(e[i], pB)
            a = R.exp(lA)
            b = R.exp(lB)
            wA = a / (a + b)
            wB = b / (a + b)
            eA = R.mul(wA, e[i])
            eB = R.mul(wB, e[i])
            sumA = R.add(sumA, eA)
            sumB = R.add(sumB, eB)

        npA = R.mul(sumA, 1.0 / np.sum(sumA))
        npB = R.mul(sumB, 1.0 / np.sum(sumB))
        dA = R.sub(npA, pA)
        dB = R.sub(npB, pB)
        delta = R.max([np.max(dA), np.max(dB)])
        pA = npA
        pB = npB

EM()
