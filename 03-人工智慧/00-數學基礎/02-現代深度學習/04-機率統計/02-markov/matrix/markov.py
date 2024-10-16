# 參考： 自然語言處理 -- Hidden Markov Model http://cpmarkchang.logdown.com/posts/192352
from prob import *

def markov(s):
    p = P[s[0]]
    for i in range(1, len(s)):
        p = p * T[s[i-1]][s[i]]
    return p

seq = [b, a, b, b]

print('P(b a b b) = P(b) P(b=>a) P(a=>b) P(b=>b) = {}*{}*{}*{} = {}'.format(P[b], T[b][a], T[a][b], T[b][b], markov(seq)))
