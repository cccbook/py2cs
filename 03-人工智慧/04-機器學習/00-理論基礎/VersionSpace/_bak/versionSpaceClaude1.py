from typing import List, Tuple, Optional

class Hypothesis:
    def __init__(self, hair: Optional[bool], eggs: Optional[bool], legs: Optional[bool]):
        self.hair = hair
        self.eggs = eggs
        self.legs = legs

    def matches(self, example: Tuple[bool, bool, bool]) -> bool:
        return (self.hair is None or self.hair == example[0]) and \
               (self.eggs is None or self.eggs == example[1]) and \
               (self.legs is None or self.legs == example[2])

    def __str__(self):
        return f"({'有毛髮' if self.hair else '無毛' if self.hair is not None else '?'}, " \
               f"{'不下蛋' if self.eggs is False else '下蛋' if self.eggs is True else '?'}, " \
               f"{'四條腿' if self.legs else '非四條腿' if self.legs is not None else '?'})"

class VersionSpace:
    def __init__(self):
        self.hypotheses = [Hypothesis(None, None, None)]

    def update(self, example: Tuple[bool, bool, bool], is_mammal: bool):
        new_hypotheses = []
        for h in self.hypotheses:
            if h.matches(example) == is_mammal:
                new_hypotheses.append(h)
            elif is_mammal:
                # 如果當前假設不匹配正例,嘗試特化
                if h.hair is None and example[0]:
                    new_hypotheses.append(Hypothesis(True, h.eggs, h.legs))
                if h.eggs is None and not example[1]:
                    new_hypotheses.append(Hypothesis(h.hair, False, h.legs))
                if h.legs is None and example[2]:
                    new_hypotheses.append(Hypothesis(h.hair, h.eggs, True))
        self.hypotheses = new_hypotheses

    def print_version_space(self):
        print("Current Version Space:")
        for h in self.hypotheses:
            print(str(h))
        print()

def main():
    vs = VersionSpace()
    
    # 訓練數據: (有毛髮, 下蛋, 四條腿, 是否哺乳動物)
    training_data = [
        (True, False, True, True),   # A
        (False, True, True, False),  # B
        (True, False, False, True),  # C
        (False, True, False, False)  # D
    ]

    print("初始 Version Space:")
    vs.print_version_space()

    for i, (hair, eggs, legs, is_mammal) in enumerate(training_data, 1):
        animal_name = chr(64 + i)  # A, B, C, D
        print(f"處理動物 {animal_name}: ({'有' if hair else '無'}毛髮, "
              f"{'不' if not eggs else ''}下蛋, "
              f"{'四條腿' if legs else '非四條腿'}) -> "
              f"{'是' if is_mammal else '不是'}哺乳動物")
        vs.update((hair, eggs, legs), is_mammal)
        vs.print_version_space()

    print("最終 Version Space (G-S 邊界):")
    vs.print_version_space()

if __name__ == "__main__":
    main()