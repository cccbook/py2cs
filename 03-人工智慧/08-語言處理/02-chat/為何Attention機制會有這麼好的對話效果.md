# 為何Attention機制會有這麼好的對話效果?

* https://www.facebook.com/ccckmit/posts/pfbid01JYyzRC1M5C5YewXWdYMuhA4HRV4vafbmj31KuFzbxi3dG31kxQaSL5ixzmoPKfzl

為何 Transformer 的 Attention Model 在自然語言對話的表現這麼好？目前我拼湊的線索如下：

1. 1964 年其實就有個 Eliza 聊天程式被創造出來，當時大家也很驚訝，但看了程式碼之後就不驚訝了，因為基本上就是兩個技巧：(1) 抓你的語句片段再回問你 (2) 用萬用語句來應付你。

2. 如果你看過《廢文產生器》之類的程式碼，會發現這些程式也很簡單，就能產生出蠻可讀的文章，基本上就是《用你標題的一部分拼湊到萬用語句中，然後再用名人語句來豐富文章結構》

3. 到了 RNN 出來之後，發現 RNN/LSTM 這些模型，可以透過《預測下一個字》，產生看來非常有模有樣的文章，例如你餵給他一堆莎士比亞，他就能寫出很像莎士比亞對話的劇本，你餵給他論文的 latex 檔，他就能寫出一堆看來很像樣的論文(還能編譯成 PDF)，你餵給他一堆 linux 的 C 語言原始碼，它寫出來的 C 語言就看來品質很好 (但實際功能不見得能跑)。這個現象在 Karpathy 的 The Unreasonable Effectiveness of Recurrent Neural Networks 這篇文章裡描述得很清楚。

4. Transformer 裡的 Attention 機制，具備了類似 RNN / LSTM 的能力，但是又加上了《注意力》可以抓取使用者意圖、語境和文字結構，於是具備了 RNN 的模仿能力，以及超越 Eliza 很多的場景意圖的能力，因此才會衍生出 GPT/ChatGPT/BERT 這樣更厲害的模型。


## 參考文獻

* https://en.wikipedia.org/wiki/ELIZA
* https://ccckmit.github.io/aibook/htm/eliza.html
* https://howtobullshit.me/
* https://github.com/telunyang/python_bullshit_generator/blob/master/Bullshit.py
* http://karpathy.github.io/2015/05/21/rnn-effectiveness/
* https://arxiv.org/abs/1706.03762




