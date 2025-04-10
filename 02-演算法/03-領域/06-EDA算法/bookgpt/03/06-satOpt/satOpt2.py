class SATSolver:
    def __init__(self, clauses):
        """
        初始化 SAT 求解器。
        
        :param clauses: CNF 格式的子句列表。例如：
                        [[1, -2], [-1, 3]] 表示 (x1 OR NOT x2) AND (NOT x1 OR x3)
        """
        self.clauses = clauses

    def solve(self):
        """
        求解 SAT 問題。

        :return: 一個可行的變數分配方案（字典形式）或 None 表示無解。
        """
        variables = {abs(literal) for clause in self.clauses for literal in clause}
        assignment = {}
        return self.dpll(self.clauses, assignment, variables)

    def dpll(self, clauses, assignment, variables):
        """
        DPLL 演算法核心。

        :param clauses: 當前的 CNF 子句。
        :param assignment: 當前變數的布爾值分配。
        :param variables: 剩餘未分配的變數集合。
        :return: 可行的分配方案或 None。
        """
        # 移除已滿足的子句
        clauses = [clause for clause in clauses if not self.is_clause_satisfied(clause, assignment)]

        # 找到空子句，無解
        if any(not clause for clause in clauses):
            return None

        # 無剩餘子句，則所有子句均滿足
        if not clauses:
            return assignment

        # 如果沒有剩餘變數，但還有子句未滿足，則無解
        if not variables:
            return None

        # 單子句（單一文字的子句），應直接分配
        for clause in clauses:
            if len(clause) == 1:
                literal = clause[0]
                var = abs(literal)
                value = literal > 0
                assignment[var] = value
                variables.discard(var)
                return self.dpll(clauses, assignment, variables)

        # 選擇一個變數嘗試分配
        var = next(iter(variables))
        variables.remove(var)

        # 嘗試將 var 分配為 True
        assignment[var] = True
        result = self.dpll(clauses, assignment, variables)
        if result is not None:
            return result

        # 嘗試將 var 分配為 False
        assignment[var] = False
        result = self.dpll(clauses, assignment, variables)
        if result is not None:
            return result

        # 恢復變數
        variables.add(var)
        del assignment[var]

        return None

    def is_clause_satisfied(self, clause, assignment):
        """
        檢查子句是否滿足。

        :param clause: 子句，包含一組文字。
        :param assignment: 當前變數分配。
        :return: True 表示滿足，False 表示未滿足。
        """
        for literal in clause:
            var = abs(literal)
            if var in assignment and assignment[var] == (literal > 0):
                return True
        return False


# 測試函數
def main():
    # 定義 CNF 格式子句。例如:
    # (x1 OR NOT x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
    clauses = [
        [1, -2],
        [-1, 3],
        [-2, -3]
    ]

    solver = SATSolver(clauses)
    solution = solver.solve()

    if solution is not None:
        print("SAT 問題有解:")
        print({f"x{var}": value for var, value in solution.items()})
    else:
        print("SAT 問題無解。")


# 執行測試
if __name__ == "__main__":
    main()
