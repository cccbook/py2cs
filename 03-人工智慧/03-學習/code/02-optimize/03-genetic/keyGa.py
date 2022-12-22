from geneticAlgorithm import GeneticAlgorithm
import random

class KeyGA(GeneticAlgorithm):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def randomChromosome(self): # 隨機產生一個染色體 (一個 16 位元的 01 字串)
        bits=[]
        for _ in range(len(self.key)):
            bit = str(random.randint(0,1))
            bits.append(bit)
        return ''.join(bits)
  
    def calcFitness(self, c): # 分數是和 key 一致的位元個數
        fitness=0
        for i in range(len(self.key)):
            fitness += 1 if c[i]==self.key[i] else 0
        return fitness
  
    def crossover(self, c1, c2):
        cutIdx = random.randint(0, len(c1)-1)
        head   = c1[0:cutIdx]
        tail   = c2[cutIdx:]
        return head + tail
    
    def mutate(self, chromosome): # 突變運算
        i=random.randint(0, len(chromosome)-1) # 選擇突變點
        cMutate = chromosome[0:i]+random.choice(['0','1'])+chromosome[i+1:] # 在突變點上隨機選取 0 或 1
        return cMutate # 傳回突變後的染色體

# 執行遺傳演算法，企圖找到 key，最多執行20代，每代族群都是一百人
kga = KeyGA("1010101010101010")
kga.run(100, 20)
