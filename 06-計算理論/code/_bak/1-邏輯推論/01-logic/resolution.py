
def neg(q):
    if q[0] == '-':
        return q[1:]
    else:
        return f'-{q}'

def unify(set1, set2):
    rset = set()
    for a in set1:
        nega = neg(a)
        if not (nega in set2):
            rset.add(a)
    for b in set2:
        negb = neg(b)
        if not (negb in set1):
            rset.add(b)
    return rset
      
def resolution(rules, q):
    nqset = {neg(q)}
    print('nqset=', nqset)
    while True:
        new_nq = False
        for nq in nqset:
            for rule in rules:
                print('nq=', nq)
                print('rule=', rule)
                r = unify(nq, rule)
                print('r=', r)
                if r:
                    print(f'{nq} is Proofed by {rule}')
                    return True # 導出空集合，代表 q 是對的
                if not r in nqset:
                    nqset.add(r)
                    new_nq = True
        if not new_nq: # 沒有產生新的 q, 結束
            break

def build_rule(rule):
    return set(rule.split("|"))
        
kb = [
    "A",
    "B",
    "-A|C",
    "-B|D",
    "-(C&D)|E",
    "-G|H"
]

rules = []

for r in kb:
    rules.append(build_rule(r))

print('rules=', rules)
resolution(rules, 'B')