from hillClimbing import hillClimbing # 引入解答類別
from solution import Solution
import random
import virginia

class SolutionVirginia(Solution):
    etext = ""

    def __init__(self, v):
        self.v = v

    def neighbor(self): # 單變數解答的鄰居函數。
        key2 = self.v.copy()
        i = random.randrange(0, len(key2))
        key2[i] = random.randint(0, 127)
        return SolutionVirginia(key2) # 建立新解答並傳回。

    def height(self):               # 能量函數
        key = self.v
        # 比對文章，看看出現多少次常用字，這就是分數
        text = virginia.decrypt(SolutionVirginia.etext, key)
        score = virginia.fit(text)
        return score

    def str(self): # 將解答轉為字串，以供印出觀察。
        return "key={} score={}".format(self.v, self.height())

    @classmethod
    def init(cls, etext, key0):
        cls.etext = etext
        return SolutionVirginia(key0)
