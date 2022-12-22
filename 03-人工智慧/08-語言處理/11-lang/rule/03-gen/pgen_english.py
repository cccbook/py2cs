import random as r

bnf = {
  'S': ['NP', 'VP'],  # S = NP VP
  'NP': ['DET', 'N'], # NP = DET N
  'VP': ['V', 'NP'],  # VP = V NP
  'N': {'words': ['dog', 'cat'], 'probs':[0.3, 0.7]},   # N = dog | cat
  'V': {'words': ['chase', 'eat'], 'probs':[0.6, 0.4]}, # V = chase | eat
  'DET': {'words': ['a', 'the'], 'probs':[0.5, 0.5]},   # DET = a | the
}

def genS():
    p = {'p': 1}
    return genRule('S', p), p['p']

def genRule(name, p):
    result = []
    rule = bnf[name]
    if isinstance(rule, list):
        for token in rule:
            result += genRule(token, p)
    else:
        result += genWords(rule['words'], rule['probs'], p)
    return result

def choose(words, probs):
    i = r.randrange(len(words))
    return (words[i], probs[i])

def genWords(words, probs, p):
    word, prob = choose(words, probs)
    p['p'] = p['p'] * prob
    return [word]

for _ in range(10):
    print('%s %f'%genS())
