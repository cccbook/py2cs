# https://chatgpt.com/c/674d3e82-0288-8012-b784-9b2c82714db5
from itertools import combinations

def to_binary(num, bits):
    """將數字轉為固定位數的二進制字串"""
    return f"{num:0{bits}b}"

def hamming_distance(bin1, bin2):
    """計算兩個二進制字串的海明距離"""
    diff = sum(1 for b1, b2 in zip(bin1, bin2) if b1 != b2)
    return diff

def combine_terms(bin1, bin2):
    """合併兩個僅有一位不同的二進制字串，使用 '-' 表示不可確定位"""
    return ''.join(b1 if b1 == b2 else '-' for b1, b2 in zip(bin1, bin2))

def is_covered(minterm, implicant):
    """檢查 minterm 是否被給定的 implicant 覆蓋"""
    for m, i in zip(minterm, implicant):
        if i != '-' and m != i:
            return False
    return True

def quine_mccluskey(minterms, dont_cares=[]):
    """
    使用 Quine-McCluskey 法簡化邏輯運算式

    :param minterms: 最小項列表
    :param dont_cares: 不關心條件列表
    :return: 簡化後的運算式
    """
    all_terms = sorted(set(minterms + dont_cares))
    bits = len(to_binary(max(all_terms), 0))  # 決定位元長度

    # 初始分組
    groups = {}
    for term in all_terms:
        ones = to_binary(term, bits).count('1')
        groups.setdefault(ones, []).append(to_binary(term, bits))

    # 簡化過程
    prime_implicants = set()
    while groups:
        next_groups = {}
        marked = set()
        for i, group in groups.items():
            if i + 1 not in groups:
                continue
            for term1 in group:
                for term2 in groups[i + 1]:
                    if hamming_distance(term1, term2) == 1:
                        combined = combine_terms(term1, term2)
                        next_groups.setdefault(combined.count('1'), []).append(combined)
                        marked.add(term1)
                        marked.add(term2)
        prime_implicants.update(set(term for group in groups.values() for term in group if term not in marked))
        groups = next_groups

    # 篩選必要項（Essential Prime Implicants）
    uncovered_minterms = [to_binary(m, bits) for m in minterms]
    essential_prime_implicants = set()
    while uncovered_minterms:
        # 計算每個 implicant 覆蓋的 minterms
        coverage = {implicant: [] for implicant in prime_implicants}
        for minterm in uncovered_minterms:
            for implicant in prime_implicants:
                if is_covered(minterm, implicant):
                    coverage[implicant].append(minterm)

        # 找出覆蓋 minterms 最多的 implicant
        most_covered = max(coverage, key=lambda imp: len(coverage[imp]))
        essential_prime_implicants.add(most_covered)
        prime_implicants.remove(most_covered)

        # 更新未覆蓋的 minterms
        uncovered_minterms = [m for m in uncovered_minterms if not is_covered(m, most_covered)]

    return essential_prime_implicants

def implicant_to_expression(implicant, variables):
    """將 implicant 轉為邏輯運算式"""
    terms = []
    for var, bit in zip(variables, implicant):
        if bit == '1':
            terms.append(var)
        elif bit == '0':
            terms.append(f"~{var}")
    return ''.join(terms)

# 測試函數
def main():
    minterms = [1, 2, 5, 6, 7]  # 最小項
    dont_cares = [3]  # 不關心條件
    variables = ['A', 'B', 'C']  # 變數名稱

    # 簡化運算
    implicants = quine_mccluskey(minterms, dont_cares)
    expressions = [implicant_to_expression(imp, variables) for imp in implicants]

    # 輸出結果
    print("簡化後的邏輯運算式:")
    print(" + ".join(expressions))

# 執行測試
main()
