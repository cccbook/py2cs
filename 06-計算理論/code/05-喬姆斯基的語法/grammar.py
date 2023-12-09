import random

def gen(G, start="S"):            # 根據語法 G 生成語句
    lefts = list(G.keys())        # 取得所有規則中的左式 (A=>B 中的 A)
    rule = f" {start} "           # 起始符號
    while True:                   # 不斷重複的生成迴圈
        for left0 in lefts:       # 對於每個左式
            left = f' {left0} '   # 在前後補上空白成為 left
            i = rule.find(left)   # 找找看目前的展開式 rule 當中有沒有這個左式 left
            if i != -1:           # 如果有找到
                rights = G[left0] # 就取得對應的右式集合 rights
                right = f' {random.choice(rights)} ' # 隨機取得右式集合裏的一個為 right
                rule = rule[0:i]+right+rule[i+len(left):] # 將左式 left 取代為右式 right
                print(rule)       # 印出目前結果以便觀察
                if rule == rule.lower(): # 如果已經全都是小寫 (非終端項目)
                    return rule   # 那就已經生成完畢，傳回！
                break             # 有生成展開的話，就 break 換新的一輪展開
