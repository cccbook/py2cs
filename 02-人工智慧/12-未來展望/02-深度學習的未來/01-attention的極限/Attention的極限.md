# Attention 的極限

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 程世嘉的貼文

* [程世嘉: OpenAI dev day 後，根據 Deep Mind 論文，討論 Attention 的極限](https://www.facebook.com/segacheng/posts/pfbid0hcavupakPKd4cFLPRaMgYrA6VcrPCDqNazs9iqE6XgiHi41o1Jp9KoVBcUJWEJ1fl)
    * DeepMind 論文 -- [Pretraining Data Mixtures Enable Narrow Model Selection
Capabilities in Transformer Models](https://arxiv.org/pdf/2311.00871.pdf)

原文：

OpenAI 在 DevDay 開發者大會中展現龐大野心，想讓 GPT Store 成為未來 AI agent 的集散地，興奮的開發者也馬上激動地把 OpenAI 類比為 AI 時代的蘋果公司，馬上就要複製當年 Steve Jobs 推出 App Store 時的成功經驗。

但我認為這個目標還遙不可及，而且發展的方式很可能會跟大家想像的完全不一樣。

話不能憑感覺亂講，我會這樣判斷主要是因為 Google DeepMind 在 DevDay 前幾天 11/3 發表了一篇論文，證實了一件 Sam Altman 可能不希望你知道太多的事情。

這篇論文大家可能看標題就直接中離了，我盡量大白話解釋，但以下完全脫離數學公式的大白話，真的很不精準，建議想深入了解細節的朋友還是直接看論文。

開始吧。

基於 Transformer 架構訓練出來的 LLM，之所以能夠展現出各種驚人的能力，主要是因為訓練資料涵蓋的範圍本來就很龐大，這份龐大資料當中已經包含各式各樣完成已知類型任務的知識，所以從「完成特定類型任務」的角度來看，這份訓練資料可以看成是一塊一塊「任務家族（task families）」聯合在一起的龐大知識庫，舉例：「寫驚悚小說」和「烘焙咖啡豆」明顯是屬於兩個不同的任務家族。

而 Transformer 擅長的事情，就是在未經任何額外訓練的狀況之下（zero shot），就可以直接理解我們現在給它的問題或要求，然後從既有的任務家族中辨識出哪個家族最適合，自己再消化並運用這個家族當中的知識（in-context learning），達成回答問題或是完成任務的目的。在 AI 發展史上，這已經是前所未有的大突破，帶起了生成式 AI 狂潮。

不過問題來了，要是訓練資料當中並不包含特定任務家族的資料，Transformer 能夠「泛化」自己既有的能力，完成前所未見的新類型任務、或是給出人類滿意的答案嗎？

根據這篇論文的研究結果，答案是否定的，Transformer 幾乎無法應付超出資料集範圍的問題或任務。
還有第二個問題：Transformer 是否擅長組合不同任務家族的技能，真正符合一個 AI agent 角色的定義，很靈活地作為我們的代理人去把事情完成呢？

根據這篇論文，答案也是否定的，Transformer 並不擅長自己拼湊出組合技。

合格的 AI agent，必須能夠找出並成功執行組合技的能力，並且要能夠機伶地調用外部工具來輔助，不然就是我一直在講的：你以為自己在做 AI agent，但你實際上是在做 RPA。

我因此相信，很多開發者興致勃勃衝到 GPT Store 開發自己的 agent 之後，會發現如果只是想要創造一個完成特定任務的 AI agent，其實走回 RPA 的老路就夠了。用 LLM 作為基底不見得是適合的，大部分的時候可能都會落入拿著鎚子到處找釘子的窘況。

這篇論文，就是用很簡約的方式驗證了這件事情。

看到這邊，你應該已經大致上明白，如果上面兩個問題的答案都是肯定的，那表示我們距離通用型人工智慧（AGI, Artificial General Intelligence，指的是跟人沒兩樣的 AI）已經不遠了，不過很遺憾，Transformer 似乎並不是人類發展通用型人工智慧一直在尋找的聖杯。

除非 OpenAI 已經掌握比 Transformer 更強大的、還未向世人發表的新 AI 基礎架構。這個新架構除了擅長組合技之外，還可以靠著外插推論出自己原本沒有的知識。但是 Sam Altman 並沒有明示或暗示 OpenAI 有更好的基礎架構，所以這個想像中的新架構很大可能是根本還不存在。

GPT Store 一開始不會像是 Apple 的 App Store 剛開始發展的榮景，馬上會有解決真實世界問題的 AI agent 大量出現。大家其實還是換湯不換藥、在 ChatGPT 既有的知識體系當中打轉。簡單來說，DevDay 發表的東西，讓 OpenAI 和開發者仍處於「將 ChatGPT 包層皮」的階段，還沒有真正在 AI agent 這個遠大目標上推進。

回歸本質，LLM 目前其實是一個龐大的知識庫，能夠當副駕駛 Copilot，但合格的 AI agent 必須是一位正駕駛。

總之，LLM 的基礎架構本身並不適合作為 AI agent 的本體，大家早就開始嘗試用 LLM 串連各種外部工具、想要建構出合格的 AI agent，但 Custom GPT 在這邊並沒有辦法幫上什麼忙（也是個噱頭，真正幫上忙的是 LangChain）。所以建構 AI agent 這個努力方向，跟 GPT Store 既沒有什麼直接關係、也沒有非要用 ChatGPT 當做基底不可。

截至目前為止，全世界建構合格 AI agent 的努力都沒有成功，包括今年上半年轟動一時的 Auto-GPT 和 AgentGPT。想要基於 LLM 打造出一個通用型的 AI agent 框架，這件事情成功率看起來不大，因為整個 Transformer 基礎架構就是不適合的。

因此 GPT Store 和 Custom GPT 在目前 AI 發展的進程上，是藉由群眾外包，看看能否真正找到一條 AI agent 可行的路，目前是在「驗證可行性」的階段，本質上與 Apple 的 App Store 完全不一樣。當然，GPT Store 或許會出現一些爆紅應用，但那些東西並不是 AI agent，在 App Store 或 Google Play 或其他地方都可以做出這些應用。

OpenAI 想要成為 AI 時代的蘋果公司，還早。


## 我的想法:

程世嘉的想法很有趣


> 根據這篇論文的研究結果，答案是否定的，Transformer 幾乎無法應付超出資料集範圍的問題或任務。

> 還有第二個問題：Transformer 是否擅長組合不同任務家族的技能，真正符合一個 AI agent 角色的定義，很靈活地作為我們的代理人去把事情完成呢？

> 根據這篇論文，答案也是否定的，Transformer 並不擅長自己拼湊出組合技。

> 合格的 AI agent，必須能夠找出並成功執行組合技的能力，並且要能夠機伶地調用外部工具來輔助，不然就是我一直在講的：你以為自己在做 AI agent，但你實際上是在做 RPA。

但因此而說：

1. 合格的 AI agent，必須能夠找出並成功執行組合技的能力，並且要能夠機伶地調用外部工具來輔助
2. LLM 的基礎架構本身並不適合作為 AI agent 的本體
3. OpenAI 想要成為 AI 時代的蘋果公司，還早。

感覺似乎過頭了
