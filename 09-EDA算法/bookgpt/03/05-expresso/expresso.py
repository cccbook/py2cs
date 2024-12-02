# https://chatgpt.com/c/674d3e82-0288-8012-b784-9b2c82714db5
class EspressoSimplifier:
    def __init__(self, num_vars):
        self.num_vars = num_vars

    def minterm_to_binary(self, minterm):
        return f"{minterm:0{self.num_vars}b}"

    def combine_terms(self, term1, term2):
        diff_count = 0
        result = []

        for b1, b2 in zip(term1, term2):
            if b1 != b2:
                diff_count += 1
                result.append('-')
            else:
                result.append(b1)

        return ''.join(result) if diff_count == 1 else None

    def find_prime_implicants(self, terms):
        """改用迭代實現以避免遞歸過深"""
        implicants = set(terms)
        while True:
            marked = set()
            new_implicants = set()

            for term1 in implicants:
                for term2 in implicants:
                    if term1 != term2:
                        combined = self.combine_terms(term1, term2)
                        if combined:
                            marked.add(term1)
                            marked.add(term2)
                            new_implicants.add(combined)

            # 保留未被合併的項
            prime_implicants = (implicants - marked) | new_implicants

            # 如果無新項產生，則停止
            if prime_implicants == implicants:
                break
            implicants = prime_implicants

        return prime_implicants

    def cover_minterms(self, minterms, implicants):
        cover = []
        uncovered = set(minterms)

        for implicant in implicants:
            covered = [minterm for minterm in uncovered if self.covers(implicant, minterm)]
            if covered:
                cover.append(implicant)
                uncovered -= set(covered)

        return cover

    def covers(self, implicant, minterm):
        for i, c in enumerate(implicant):
            if c != '-' and c != minterm[i]:
                return False
        return True

    def simplify(self, minterms, dont_cares):
        terms = [self.minterm_to_binary(m) for m in minterms + dont_cares]
        prime_implicants = self.find_prime_implicants(terms)
        essential_prime_implicants = self.cover_minterms(
            [self.minterm_to_binary(m) for m in minterms],
            prime_implicants
        )
        return essential_prime_implicants


# 測試函數
def main():
    num_vars = 3
    minterms = [1, 2, 5, 6, 7]
    # dont_cares = [3]
    dont_cares = []

    simplifier = EspressoSimplifier(num_vars)
    simplified = simplifier.simplify(minterms, dont_cares)

    print("簡化後的布林代數式:")
    for term in simplified:
        print(term)


# 執行測試
main()
