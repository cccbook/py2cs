def satisfy(exp, facts): # 檢查 facts 是否滿足邏輯式 exp
    vars = exp.split("&")
    for var in vars:
        if not var.strip() in facts:
            return False
    return True

def gen(facts, rules): # 嘗試推論出新的事實
    gen_new = False
    for rule in rules:
        left, right = rule.split("=>")
        if satisfy(left, facts):
            gens = right.split("&")
            for g in gens:
                if not g in facts:
                    print("facts=", facts)
                    facts.add(g)
                    print(f"inference:\n\t{left}\n\t{rule}\n\t-------\n\t{g}\n\n")
                    gen_new = True

    return gen_new

def inference(facts, rules): # 推論引擎
    i=1
    while True:
        print(f"=========gen {i}============")
        if not gen(facts, rules): # 反覆推論，直到失敗為止
            print('沒有新的結果產生 ... ，推論完畢！')
            break
        i = i+1

def load(file):
    facts = set()
    rules = []

    with open(file, "r", encoding="utf-8") as f:
        text = f.read().replace(" ", "")

    for line in text.split("\n"):
        if line.find("=>") != -1:
            rules.append(line)
        elif len(line)>0:
            facts.add(line)
            
    return facts, rules

import sys
facts, rules = load(sys.argv[1]) # 載入知識庫
inference(facts, rules) # 開始推論
