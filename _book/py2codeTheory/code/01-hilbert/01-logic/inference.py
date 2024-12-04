def satisfy(exp, facts): # 檢查 facts 是否滿足邏輯式 exp
    vars = exp.split("&")
    for var in vars:
        if not var in facts:
            return False
    return True

def dump(facts, rules): # 印出事實與規則庫
    print('facts:', facts)
    print('rules:', rules)

def gen(facts, rules): # 嘗試推論出新的事實
    gen_new = False
    for rule in rules:
        left, right = rule.split("=>")
        if satisfy(left, facts):
            gens = right.split("&")
            for g in gens:
                if not g in facts:
                    dump(facts, rules)
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

facts = {"A", "B"} # 事實
rules = [ # 規則庫
    "E&F=>H",
    "A=>E",
    "C=>G",
    "B=>F",
]

inference(facts, rules) # 開始推論

