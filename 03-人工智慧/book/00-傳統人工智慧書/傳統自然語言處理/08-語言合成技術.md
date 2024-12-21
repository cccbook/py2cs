## 語言合成技術

### 語言合成的範例 -- SciGen

SciGen 是一個自動產生論文的程式，該程式由美國麻省理工學院計算機科學與人工智慧實驗室的三位研究生傑里米·斯特里布林（Jeremy Stribling）、馬克斯·克倫（Max Krohn）和達納·阿瓜約（Dan Aguayo）所設計。

SciGen 產生的論文曾經被拿去大量投稿到期刊與研討會，結果還有一些論文被接受準備刊登了，關於 SciGen 的進一步歷史請參考：

* 參考 -- https://zh.wikipedia.org/wiki/SCIgen

SciGen 有線上版，您可以實際使用 SciGen 產生論文看看：

* 系統 -- 用 SciGen 示範如何自動產生論文 (這個版本似乎會快取)
    * https://pdos.csail.mit.edu/archive/scigen/

### 語言合成的方法

其實語言合成只要會使用 遞迴呼叫的方式，撰寫 BNF 語法的生成程式就行了，您可以試著做做下列習題：

習題 1 ：自動產生英文語句

提示：先用簡單的幾個字加上基本語法就行了，不用一下企圖心太大。

簡易英文語法

```
S = NP VP
NP = DET N
VP = V NP
N = dog | cat
V = chase | eat
DET = a | the
```

產生過程的範例

```
S = NP VP = (DET N) (V NP) 
  = (a dog) (chase DET N) 
  = a dog chase a cat
```

程式的使用範例

```
PS D:\ccc\book\aijs\code\07-language> node genEnglish
a cat eat the dog
PS D:\ccc\book\aijs\code\07-language> node genEnglish
the cat chase the dog
PS D:\ccc\book\aijs\code\07-language> node genEnglish
a cat chase a cat
```

習題 2 : 自動產生運算式語句

簡易運算式語法

```
E = T [+-*/] E | T
T = [0-9] | (E)
```

程式執行結果：

```
nqu-192-168-61-142:code mac020$ node genexp
4
0/0+(2)*9
4-(9)*((((3*(4))-(8))+(0)+8/(8)+2)+2/6)
3/(((((1*8+6)))))*((6/4/3))/(((2+9))+(((2))+8/((4*(5))*2))/4)
(1+(1))-((7))
(2+(((4))))+(5)
((1/(((3+(7)-(4-1)/9*8/7-6)/(4)-3+3)-6-9*(((2+(((6*4/4)))*(8/3))))-9-0-1+5*8*((5)/(3)-1/(1)-9)+(5+5*5))))*5/2
8
1
(0)*7
```

習題 3 : 小學數學問題產生器

```
S D:\ccc\book\aijs\code\07-language> node genMath
問題:   大雄有12個番茄
        給了小華9個
        又給了小華2個
        請問大雄還有幾個番茄?

答案:   1個
PS D:\ccc\book\aijs\code\07-language> node genMath
問題:   小明有17個蘋果
        給了小莉16個
        又給了小莉1個
        請問小明還有幾個蘋果?

答案:   0個
PS D:\ccc\book\aijs\code\07-language> node genMath
問題:   小華有6個番茄
        給了大雄3個
        又給了小明2個
        請問小華還有幾個番茄?

答案:   1個
```

