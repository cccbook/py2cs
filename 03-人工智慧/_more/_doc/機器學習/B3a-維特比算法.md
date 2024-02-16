## 維特比演算法（Viterbi algorithm）


維特比演算法是高通創辦人 Viterbi 所設計的一個方法，原本是用來去除通訊系統雜訊用的，後來在《語音辨識與自然語言處理領域》也很常被使用，因為維特比演算法可以很快地計算《隱馬可夫模型》的最可能隱序列。

關於《隱馬可夫模型》與《維特比演算法》的說明，請參考下列文章：

* [維基百科:維特比演算法](https://zh.wikipedia.org/wiki/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95)

《維特比演算法》是用來尋找產生某『表現序列』的最可能『隱狀態序列』，以下是我們用程式 Viterbi.js 計算的結果：

範例：根據下列規則，請問『喵 喵 汪』中每個詞彙最可能的詞性會是什麼？

轉移機率與規則

```
N 0.6 => 喵 0.4 | 汪 0.6
V 0.4 => 喵 0.5 | 汪 0.5

    N   V
 N  0.3 0.7
 V  0.8 0.2
```

執行結果

```
$ python3 viterbi.py
觀察到的序列= ['喵', '喵', '汪']
T= [{}, {}, {}, {}]
t=1 path={'N': ['V', 'N'], 'V': ['N', 'V']}
t=2 path={'N': ['N', 'V', 'N'], 'V': ['V', 'N', 'V']}
T= [{'N': 0.24, 'V': 0.2}, {'N': 0.06400000000000002, 'V': 0.08399999999999999}, {'N': 0.040319999999999995, 'V': 0.022400000000000003}, {}]
prob=0.040319999999999995 path=['N', 'V', 'N']＝最可能的隱序列
```

以下是該程式的原始碼：

```py
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

```

### 觀念

* [維基百科:維特比演算法](https://zh.wikipedia.org/wiki/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95)
* [自然語言處理 -- Hidden Markov Model](https://ckmarkoh.github.io/blog/2014/04/03/natural-language-processing-hidden-markov-models/)
* [自然語言處理 -- Viterbi Algorithm](https://ckmarkoh.github.io/blog/2014/04/06/natural-language-processing-viterbi-algorithm/)

### 實作

* https://github.com/miguelmota/hidden-markov-model
* https://github.com/123jimin/hmm.js/blob/master/hmm.js
* https://github.com/123jimin/hmm.js/tree/master


### 參考文獻

* [維基百科:維特比演算法](https://zh.wikipedia.org/wiki/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95)
