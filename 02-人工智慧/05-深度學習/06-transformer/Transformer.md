# Transformer

## seq2seq model

seq2seq model : 輸入長度不固定，輸出長度不固定

例如：

* 語音辨識
* 翻譯
* 自動摘要
* 台語音轉中文
* 語法剖析與標記 (輸出語法樹，語意樹，用 LISP 的語法 encode)
* Multi-Label Classification (多重分類)
* Q&A

其中大部分應用都可以 reduce 成 Q&A 的問題

Q&A transformer 就像一把瑞士刀，甚麼都能做，但不一定好用。

## Model

* Encoder-Decoder
    * 其中 Encoder 是 Attention，Decoder 是 Masked Attention
    * Decoder 每個輸出都是大向量 (例如中文字，5000 維向量，代表輸出該字的機率，通常取最大的為輸出)
    * AT: AutoRegressive (自回歸)， 加上 Input BEGIN (`<BOS>`)， Output END (`<EOS>`)
    * NAT: NonAutoRegressive (非自回歸)，好處是可平行化，一次產生整排，Decoder 的輸入均為 BEGIN，但先 Learn 一個 Classifier 去決定長度。(另一種做法是直接假設長度上限)
