import random
import math

class GeneticAlgorithm:
    def __init__(self): 
        self.population = []    # 族群
        self.mutationRate = 0.1 # 突變率

    def run(self, size, maxGen) :  # 遺傳演算法主程式
        self.population = self.newPopulation(size) # 產生初始族群
        for t in range(maxGen):  # 最多產生 maxGen 代
            print("============ generation", t, "===============")
            self.population = self.reproduction() # 產生下一代
            self.dump() # 印出目前族群
  
    def newPopulation(self, size): 
        newPop=[]
        for _ in range(size): 
            chromosome = self.randomChromosome() # 隨機產生新染色體
            newPop.append({'chromosome':chromosome, 
                           'fitness':self.calcFitness(chromosome)})
        newPop.sort(key = lambda c: c['fitness']) # 對整個族群進行排序
        return newPop
  
    # 輪盤選擇法: 隨機選擇一個個體 -- 落點在 i*i ~ (i+1)*(i+1) 之間都算是 i
    def selection(self): 
        n = len(self.population)
        shoot  = random.randint(0, (n*n/2)-1)
        select = math.floor(math.sqrt(shoot*2))
        return self.population[select]
  
    # 產生下一代
    def reproduction(self):
        newPop = []
        for i in range(len(self.population)): 
            parent1 = self.selection()['chromosome'] # 選取父親
            parent2 = self.selection()['chromosome'] # 選取母親
            chromosome = self.crossover(parent1, parent2) # 父母交配，產生小孩
            prob = random.random()
            if prob < self.mutationRate: # 有很小的機率
                chromosome = self.mutate(chromosome) # 小孩會突變
            newPop.append({ 'chromosome':chromosome, 'fitness':self.calcFitness(chromosome) })  # 將小孩放進下一代族群裡
        newPop.sort(key = lambda c: c['fitness']) # 對新一代根據適應性（分數）進行排序
        return newPop
  
    def dump(self):  # 印出一整代成員
        for i in range(len(self.population)):
            print(i, self.population[i])

