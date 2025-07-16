import random as r

'''
S => NP VP	          句子 = 名詞子句 接 動詞子句
NP => Det Adj* N PP*  名詞子句 = 定詞 接 名詞 接副詞子句
VP => V NP       	  動詞子句 = 動詞 接 名詞子句
PP => P NP	          副詞子句 = 副詞 接 名詞子句
N = 狗 | 貓
V = 追 | 吃
DET = 一隻 | 這隻
Adj = 白 | 黑 | 兇 | 帥
P = 那個 | 有 
'''

def S():
    return NP() + ' ' + VP()

def star(NT):
	list = []
	while r.choice([0,1])==0:
		list.append(NT())
	return ' '.join(list)

# NP => Det Adj* N                // PP*
def NP():
    return DET() + ' ' + star(ADJ) + ' ' + N() # + ' ' + star(PP)

def VP():
    return V() + ' ' + NP()

# PP => P NP	          副詞子句 = 副詞 接 名詞子句
def PP():
	return P() + ' ' + NP()

def P():
    return r.choice(['那個', '有'])

def N():
    return r.choice(['狗', '貓'])

def V():
    return r.choice(['追', '吃'])

def DET():
    return r.choice(['一隻', '這隻'])

# Adj = 白 | 黑 | 兇 | 可愛的
def ADJ():
	return r.choice(['白', '黑', '兇', '帥'])

print(S())
