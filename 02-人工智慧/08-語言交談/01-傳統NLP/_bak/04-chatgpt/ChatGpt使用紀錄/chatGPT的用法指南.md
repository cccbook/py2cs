# chatGPT 的用法指南

## 已經在網路上找到程式，但看不懂

* [ccc: 用 ChatGPT 幫助你學習不會的程式](https://www.facebook.com/ccckmit/posts/pfbid07YgMSwRSxUw84T1pNfy94rUMGhYdmk4etakqFWeAtEXxoXmDMWKxCovQAj8hxbCTl)

對學程式的人，可以用 ChatGPT 幫助你學習不會的程式，方法如下：
1. 首先找一個你不會的主題，例如 《受限波茲曼機的 CD-K 演算法》，我就先 Google : RBM cd-k python
2. 找到一個範例程式後，執行看看 (必要時修改一下) 
例如我找到 https: //gist .github. com/ yusugomori/4428308 
然後將其 python 2 的 print 語法改為 python 3
3. 把該程式貼到 ChatGPT 請他解釋
4. 對於 ChatGPT 沒說清楚的部分，進一步追問
例如: 
* 那函數contrastive_divergence的原理呢?
* 如何模擬 RBM 模型的推論過程?
* 請說明 sample_v_given_h 函數的原理

這種方法非常好用，我用它來學《受限波茲曼機的 CD-K 演算法》的過程，請看回應連結！


## 讓 ChatGPT 教你程式

ChatGPT 完全可以當我老師，教我程式了：

我請他教我 EM 演算法，一開始他給我 scikit 的範例，但沒給 data X，於是我再問他如何產生 data X ，經過兩三次問答，就完成了程式

ChatGPT 真的強，真的強，真的強，太強了 ...

我和 ChatGPT 的完整對話在此

https://gitlab.com/cccnqu111/alg/-/blob/master/04a-iterative/EM/chatGPT/emScikit1.md

## 讓 ChatGPT 寫程式，要小心

ChatGPT 回答的 30 行以下的程式，通常可以正確執行。
超過 70 行的程式通常會有錯 ...

## 可以丟程式文章給 ChatGPT 看過，然後問他問題

* https://www.facebook.com/ccckmit/posts/pfbid02i4AuCmumuyvVAuzXM1eyJK94kT2RtMesTkQg6oGsbkuGFya4jV6m4uScaYiaUpfRl

ChatGPT 真的超厲害！
聽說可以丟文章給他學
於是我把自己的 mini-riscv-os 第一篇文章丟給他，他就學會了 ...
請看留言中的貼文，有 ChatGPT 看我的文章後學會 RISC-V 程式的案例

## 允許學生用 ChatGPT 交作業

* https://www.facebook.com/ccckmit/posts/pfbid02kemk6Fgz8qTquzhroeN26inU1JcWZRGyEaiRxZdjZoKdUX1DqhYMoznRAtCx6S1Jl

決定了，這學期的作業我將允許學生們用 ChatGPT 來交，但必須要誠實宣告以下情形：
1. 我用 ChatGPT 產生後直接交，但我沒有看懂
2. 我用 ChatGPT 產生後直接交，我看懂了，但沒有修改
3. 我用 ChatGPT 產生後，看懂了，修改格式後就繳交了
4. 我用 ChatGPT 產生後，看懂了，並且做了修正，然後繳交。
5. 作業是我寫的，但 某某部分不會，於是問 ChatGPT ，然後看懂了之後，修改了那些部分，才繳交給老師。
6. 完全是原創，沒有問 AI 也沒有剪貼自網路。
新時代，看來得有新做法，但重點是，必須交代清楚，哪些是你寫的，哪些是 ChatGPT 或網路上來的。

## 用 Google 驗證 ChatGPT

* https://www.facebook.com/ccckmit/posts/pfbid037w4jz3LvQhoXe4vhjwc3DXC3CiDjSW6MMDWLp6TgsPNZDx3wGA4pF8wRCyr53uX9l

有了 ChatGPT 之後，Google 還是很重要

因為 ChatGPT 常常會說錯，甚至會騙人，這時候你就得靠 Google 去查查看他說的是否有問題了 ...

## 會說錯與騙人的 ChatGPT

* https://www.facebook.com/ccckmit/posts/pfbid0QVng27uYJR6kg7WQ68f5uPaMapcbDj7fHzweFmEn9AGyCBCNq9aC1Bh1hMJHnQoGl

很多人覺得 ChatGPT 會記錯，有時候會講錯，不夠好！

但我覺得，這正是 ChatGPT 這個技術有趣的地方，更有人性，和人一樣會記錯腦補，這不是很可愛嗎？

你跟他說他錯了之後，他還會修正，然後說:

> 您是對的，抱歉提供了錯誤的信息。 ...

## 寫程式書並不好用，若先有程式 ChatGPT 可以寫註解與文章

https://www.facebook.com/ccckmit/posts/pfbid0qoaAMDBdReYVi7h4TEJxRcuP9V3W3Mg7THRtt6VPhdihwkDQVZzd97nbFiavfP4El

我昨天試過用 ChatGPT 來寫整本的程式書，並不好用，我還得幫他的程式除錯，花時間太多 ...
我改變方法，把我的程式給他寫註解，好用很多，ChatGPT 寫得又快又好，(很少有錯，試了快十個，目前只看到一個有錯，一個解釋不夠好)

## ChatGPT 會有重大影響嗎?

https://www.facebook.com/ccckmit/posts/pfbid0CPgByAiToupy8ndwiY77ZcEvJB2zJhGxSkuBBwe9YRDgTz3iUbmSgQn3xajKngqcl

歷史上有些看來不重要的技術，最後產生很大的影響，像是：

> 馬鐙，條碼，短訊系統 ...

但也有不少看來很重要的技術，卻遲遲無法發生影響力，像是:

> 人工智慧系統 (專家系統，機器翻譯，Siri ....)，登陸月球，火星...

ChatGPT 這次的突破，不知道會不會變成後者 ...

* https://www.facebook.com/ccckmit/posts/pfbid02gufQWav8NjWqBsYYmKi6m2x7B3fTqTjFR2VsaCcysTzUXiaafG843b9K4C9jJPcPl

這感覺蠻有道理了，程序員會失業，黑手不會失業 ....

## Python 呼叫 ChatGPT 

https://www.facebook.com/ccckmit/posts/pfbid0M4b9DguinxRjJr9iAeh9UxhYD62QxoAH27MtYiuxVgmvsyLrGuCTcDcTbUdsGjzbl

想用 Python 去呼叫 ChatGPT 的人，可以看回應的貼文連結，有程式與用法

## 檢查語法錯誤

* https://www.facebook.com/ccckmit/posts/pfbid029SDTVkt8qfNHSt87XK6qy1sC81uAwSyA7iDmvaahamcZAzvUfWk6c5UzMmVTwcusl

## 法律才是 ChatGPT 最大的障礙

* https://www.facebook.com/ccckmit/posts/pfbid02dDHQ4kNXzCdQ1Zt1uhh3d2stiwrUFwum1mUsFpXdwH4v8xdQxBrJuDsqUJFNAc1kl

經過這幾天的使用，ChatGPT 已經證明了具有商業價值，但是最大的問題不在技術
法律才是 ChatGPT 最大的障礙！
如果你用 ChatGPT 產生出來的文章，會被法官判侵權，然後向你和 openai 求償，那你還敢用嗎？ OpenAI 不會被告到把 ChatGPT 下架嗎？
在美國的歷史上，這樣的法律問題非常多，最後結果很難預料，有興趣的人可以看看《自由文化》(Free Culture) 這本書。
雖然 github 已經把 co-pilot 這樣的軟體商業化，這和 ChatGPT 有點像，但是 co-pilot 鎖定在《開放原始碼》程式，讓問題減低不少。
ChatGPT 的訓練語料，能否鎖定在《創作共用，開放原始碼和公共領域作品》，或許會是一個關鍵。
但既使如此，我們還是很難知道法庭上會如何判定，這還得等未來案例產生後才知道！

## ChatGPT 可以看懂很抽象的歌詞

https://www.facebook.com/ccckmit/posts/pfbid0RU2nAVbUQgzEaLayb5AJxqapGPvtvZXbd1ZK6xNMQ2gSjKvx5tsj24ZrNgeRbUhvl

25 年前我做自然語言處理，最後因為周杰倫的一首歌而放棄了這個領域，那就是菊花台
---
妳的淚光 柔弱中帶傷 慘白的月彎彎勾住過往
夜太漫長 凝結成了霜 是誰在閣樓上冰冷的絕望
雨輕輕彈 朱紅色的窗 我一生在紙上被風吹亂
夢在遠方 化成一縷香 隨風飄散妳的模樣
菊花殘 滿地傷 你的笑容已泛黃
花落人斷腸 我心事靜靜躺
北風亂 夜未央 你的影子剪不斷
徒留我孤單 在湖面 成雙
花已向晚 飄落了燦爛 凋謝的世道上命運不堪
愁莫渡江 秋心拆兩半 怕妳上不了岸一輩子搖晃
誰的江山 馬蹄聲狂亂 我一身的戎裝呼嘯滄桑
天微微亮 妳輕聲的嘆 一夜惆悵如此委婉
菊花殘 滿地傷 你的笑容已泛黃
花落人斷腸 我心事靜靜躺
北風亂 夜未央 你的影子剪不斷
徒留我孤單 在湖面 成雙
菊花殘 滿地傷 你的笑容已泛黃
花落人斷腸 我心事靜靜躺
北風亂 夜未央 你的影子剪不斷
徒留我孤單 在湖面 成雙
----
我當時覺得，AI永遠都不可能讀懂菊花台這首歌！
現在 ChatGPT 完成了我年輕時的夢想，我當然要給他這首歌，看看他懂多少?

## 感想

我 25 歲的時候，就夢想著能做出像 ChatGPT 這樣的系統
我 35 歲的時候，知道自己做不到，而且認為 50 年內都沒人能做到 ...
當我 53 歲的時候，ChatGPT 做到了，雖然我沒有任何貢獻，但我很高興能成為 ChatGPT 的使用者，親眼看到年輕時的夢想被實現了 ...

## 向 chatGPT 學習 xv6 作業系統

https://www.facebook.com/ccckmit/posts/pfbid0LTCsC2Nk12AHmfHQAUkpAuVQ2RuAKJzfqaVjeE39Cu24SL7GxqZ3cwy86bCuKVAql

## ChatGPT 教我甚麼是 GPT/BERT/Transformer ...

https://www.facebook.com/ccckmit/posts/pfbid02TSNdvhYSpTLxtKK6XzBiz5gDbYSs4P3sVTzSa8xSjA3nYK1omtEGEbNEqTC217Pxl

## ChatGPT 會修正程式

https://www.facebook.com/ccckmit/posts/pfbid02GT2JShGd1xPLP3nZAftCLNsBZJNacEcjYPUmCeGPhfSfjEDAgWwDiJy6mx1r4q9ql

## ChatGPT 教你專業知識

* https://www.facebook.com/ccckmit/posts/pfbid02Sv1jWMj6jTL1E24hCccVZjPaZB4fFP7abXmDFm7LqNownns474VZJhyToN6J3Q1rl

以後別再萬事問網友了，問 ChatGPT 的回答品質好得多啊 ...
ChatGPT 教我 IC 製程的知識 ...

## ChatGPT 有時寫程式會拿香蕉比雞腿

https://www.facebook.com/ccckmit/posts/pfbid037kudwxWnEHdPGenytAXTcDxgBVtep5R2QrMDxarGEcoEFwPBV9VcZP7mSE3kqESEl

```
用了 ChatGPT 之後，我開始懷疑人生了？
ccc: 給個 python 的貝氏網路範例吧！
ChatGPT:
import numpy as np
# 定義貝氏網路的權重和偏差
weights = np.array([0.1, 0.2, 0.3])
bias = 0.1
# 定義輸入
inputs = np.array([0.5, 0.3, 0.7])
# 計算輸出
output = np.dot(weights, inputs) + bias
# 輸出結果
print(output)
等等，這到底是不是貝氏網路阿?
```

## ChatGPT 有基本的邏輯推理能力

https://www.facebook.com/ccckmit/posts/pfbid02u8ewgJf1MNp71GfppUp7aYP8FJHPQC1d1ynAgNJfmHCdgokmkyMPZevBPTrkcYTMl

## ChatGPT 不知道時會自己腦補

* https://www.facebook.com/ccckmit/posts/pfbid02rvDv4jSwkMmkUKmknM6EsjXWht1NReF8NgVtyVfa5TXmKx9FFqmGenrM7mxhFEvgl

原來可以坐火車從金門到廈門了

## ChatGPT 會證明數學定理

我請 ChatGPT 證明歐拉公式，結果回應如下

* https://www.facebook.com/ccckmit/posts/pfbid0a5LgM6F4v1hZpyuev4wMFh8wLJgr81QU4AzbJDQYPVdWiWJdvUDTHcrig6UhKVh9l