# 參考： 自然語言處理 -- Hidden Markov Model http://cpmarkchang.logdown.com/posts/192352
from prob import P
def markov(s):
    p = P[s[0]]
    for i in range(len(s)-1):
        key = f'{s[i]}=>{s[i+1]}'
        p = p * P[key]
    return p

seq = ['b', 'a', 'b', 'b']

print(f'P({seq})={markov(seq)}')
