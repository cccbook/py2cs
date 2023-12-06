from random import *

def gen(G):
    lefts = list(G.keys())
    rule = "S" # 起始符號固定為 S
    while True: # 不斷展開的回圈
        for left in lefts: # 想辦法把某個左式 left 展開成右式 right
            i = rule.find(left) # 從目前展開式 rule 中找出左邊的式子 left 是否有出現過
            if i != -1:
                rights = G[left] # 取得對應的右邊式子集合 rights
                ri = randrange(0, len(rights)); right = rights[ri] # 選取其中一個
                rule = rule[0:i]+right+rule[i+len(left):] # 將左式 left 取代成右式 right
                print(rule) # 印出新的中間展開式
                if rule.islower(): return rule # 如果全是小寫（非終端項目），代表展開完畢，傳回
                break
