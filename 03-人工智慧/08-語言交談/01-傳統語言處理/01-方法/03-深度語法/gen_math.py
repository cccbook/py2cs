'''
問題:   小華有6個蘋果
        給了大雄3個
        又給了小明2個
        請問小華還有幾個蘋果?

答案:   1個
'''
import random

peoples = ["小明", "小華", "小莉", "大雄"]
objs = ["蘋果", "橘子", "柳丁", "番茄"]

def People():
    return random.choice(peoples)

def Object():
    return random.choice(objs)

owner = People()
obj = Object()
nOwn = random.randint(3, 20)

def MathTest():
  return "問題:\t"+Own()+"\n\t"+Give()+"\n\t又"+Give()+"\n\t"+Question()

def Own():
    return owner+"有"+str(nOwn)+"個"+obj

def Give():
    global nOwn
    nGive = random.randint(1, nOwn)
    nOwn-=nGive
    return "給了"+People()+str(nGive)+"個"

def Question():
    return "請問"+owner+"還有幾個"+obj+"?"

peoples.remove(owner)
print(MathTest())
print("\n答案:\t"+str(nOwn)+"個")