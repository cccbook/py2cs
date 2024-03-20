'''
參考： https:#zh.wikipedia.org/wiki/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95
N 0.6 => 喵 0.4 | 汪 0.6
V 0.4 => 喵 0.5 | 汪 0.5
   N   V
N  0.3 0.7
V  0.8 0.2
'''

P = {
  'N': 0.6,
  'V': 0.4,
  'N=>N': 0.3,
  'N=>V': 0.7,
  'V=>N': 0.8,
  'V=>V': 0.2,
  'N=>喵': 0.4,
  'N=>汪': 0.6,
  'V=>喵': 0.5,
  'V=>汪': 0.5,
}

def argmax(alist):
    max = -999999
    index = None
    for k in range(len(alist)):
        if alist[k] > max:
            index=k
            max=alist[k]
    return max, index

def viterbi(obs, states, P):
    print('觀察到的序列=', obs)
    T = [{} for _ in range(len(obs)+1)] # [{}]*(len(obs)+1) # Viterbi Table
    print('T=', T)
    path = {}  # path[state] = 從 0 到 t 到達 state 機率最大的 path

    for y in states: # Initialize base cases (t == 0)
        T[0][y] = P[y] * P[y+'=>'+obs[0]]
        path[y] = [y]

    for t in range(1, len(obs)): # Run Viterbi for t > 0
        newpath = {}
        for y in states:
            prob, si = argmax(list(map(lambda y0:T[t-1][y0] * P[y0+'=>'+y] * P[y+'=>'+obs[t]], states)))
            state = states[si]
            T[t][y] = prob
            newpath[y] = path[state] + [y] # concat(path[state], y)
        path = newpath
        print('t={} path={}'.format(t, path))

    prob, si = argmax(list(map(lambda y:T[len(obs) - 1][y], states)))
    print('T=', T)
    return [prob, path[states[si]]]

prob, path = viterbi('喵 喵 汪'.split(' '), ['N', 'V'], P)
print('prob={} path={}＝最可能的隱序列'.format(prob, path))
