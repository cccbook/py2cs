# 後 LLM 時代

在人文社會領域，如果一個思想 X 太過有主導力，那麼這個思想之後，受該思想影響卻又有些新意的學說，通常都會被歸類為《後 X 思想》

* [后XX时代/主义是什么意思？](https://www.zhihu.com/question/308916137)

例如：

1. [後現代主義](https://zh.wikipedia.org/zh-tw/%E5%90%8E%E7%8E%B0%E4%BB%A3%E4%B8%BB%E4%B9%89)
2. [後馬克思主義](https://wiki.mbalib.com/zh-tw/%E5%90%8E%E9%A9%AC%E5%85%8B%E6%80%9D%E4%B8%BB%E4%B9%89)
3. [後結構主義](https://zh.wikipedia.org/zh-tw/%E5%BE%8C%E7%B5%90%E6%A7%8B%E4%B8%BB%E7%BE%A9)

對於程式人，我們正處於這樣一個時代

我稱之為《後 LLM 時代》 ...

## LLM 大型語言模型

2022 年 ChatGPT 出現之後， AI 領域的研究，變得完全不同了 ...

雖然 GPT 只是 2018 年 Attention is all you need 這篇論文的一個變種，但卻是真正將語言模型 (Language Model, LM) 落實為應用程式的一個神經網路模型。

由於 ChatGPT 背後的 GPT 3.5 語言模型相當龐大 (神經網路參數很多) ，因此這類的語言模型就被稱為大型語言模型 (Large Language Model, LLM)

於是 2022 到 2024 年，AI 研究者忙著探索 LLM 為何有效，如何改進，可以怎麼用？ 而那些資訊工程系的學生，也一窩蜂地投入到 AI 領域 

然後，在 Facebook 主導下出現了 LLAMA / LLAMA2 ，在 Google 主導下出現了 Bard / Gemini  ....

其實在 LLM 之前，AI 領域其實就有明顯的進展，這些進展被人們用《深度學習》一詞來代表 ...

在 LLM 之後，深度學習一詞又逐漸被搶掉鋒芒，凡是和 LLM 無關的，似乎就都不重要了 ...

LLM 令人驚豔的地方有很多，像是：

1. 可以交談
2. 可以當翻譯器
3. 你只要告訴他要寫甚麼程式，他就幫你寫出來了
4. 可以角色扮演
5. ...

後來， Diffusion Model (散射模型) 的發展，讓 LLM 的能力延伸到影音，於是有了

1. text2image : 你告訴他畫甚麼，他就幫你畫出來 (Bing Image Creator / DALL-E 3 / Midjourney / ...)
    * https://www.bing.com/images/create
    * [11 Best AI Image Generators in 2024](https://visme.co/blog/best-ai-image-generator/)
2. text2video : 你輸入劇本，他就幫你拍出影片 (Lumen5 / InVideo / ...)
    * https://lumen5.com/
    * https://openai.com/sora
    * [10 Best Tools for Text2Video By AI Generators](https://falahgs.medium.com/10-best-tools-for-text2video-by-ai-generators-47608e223948)
3. text2music : 你告訴他想做甚麼風格的歌曲，他就幫你做好了 (Suno / AIVA)
    * https://www.suno.ai/
    * https://www.aiva.ai/

## 後 LLM 時代的程式

如上所述， 深度學習+LLM 做到了很多原本程式設計者很難做到的事情，像是：

1. 影像辨識
2. 影像生成
3. 語音辨識
4. 語音生成
5. 圍棋下得很好
6. 像人類一樣《閱讀、理解、交談、回應、整理知識、吸收後內化》 ...

雖然 LLM 很強大，但是也有弱點，例如：

1. 會亂講話 :  (被稱為 LLM 幻覺) 基本上 LLM 的運作機制就是個接龍遊戲，當無法接又要硬接的時候，就會亂講
    * [大模型的幻觉问题调研: LLM Hallucination Survey](https://zhuanlan.zhihu.com/p/642648601)
2. 有些事情做不好： 例如計算很大數字的加減乘除 (傳統確定性算法就能做好的事情，LLM 經常反而做得不好)
    * Karpathy 說這些 LLM 做不好的事情，似乎需要一個 System2 來處理。
    * [Andrej Karpathy on Youtube: Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)
3. 記憶力很短： 由於 LLM API 通常沒有 session 機制，而且 LLM 在回答之前通常不會先回憶和你說過些甚麼話，因此常常會有《失意》的情況
    * [How to keep session with gpt-3.5-turbo api?](https://community.openai.com/t/how-to-keep-session-with-gpt-3-5-turbo-api/81029/7)

於是在《後 LLM 時代》，人們開始尋求各種方法，來《彌補、強化、治療、調教、整合、發揮》 LLM 的能力，這些技術包含


1. 提示工程: (Prompt Engineering)
    * 搞懂如何下提示給 LLM ，才能得到好的回應？
2. RAG: (Retrieval Augomanted Generation) 先檢索、閱讀後再回應
    * [What Is Retrieval-Augmented Generation, aka RAG?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)
    * 使用 [向量資料庫](https://aws.amazon.com/tw/what-is/vector-databases/) 進行語意檢索
3. ReAct: 
    * https://promptingguide.azurewebsites.net/techniques/react
    * https://www.promptingguide.ai/techniques/react
    * [How Agents for LLM Perform Task Planning 大型語言模型的代理如何進行任務規劃](https://hackmd.io/@YungHuiHsu/rkK52BkQp?utm_source=preview-mode&utm_medium=rec)
4. COT: Chain of Thoughts
    * https://promptingguide.azurewebsites.net/techniques/cot
    * [ReAct, Chain of Thoughts and Trees of Thoughts explained with example](https://medium.com/data-science-in-your-pocket/react-chain-of-thoughts-and-trees-of-thoughts-explained-with-example-b9ac88621f2c)
5. Graph Prompting
    * https://www.promptingguide.ai/techniques/graph

對於寫程式的人來說，在後 LLM 時代，我所關注的問題是

1. 既然 LLM 已經會寫程式了，那還有多少程式是要人類來寫的呢？
2. 對寫程式的人而言， 如何才能善用 LLM 的能力，寫出更好的程式？
3. 對程式新手而言，接下來該怎麼學程式，才能面對未來的環境呢？

到目前為止，這些問題都還處於一團迷霧當中，就像賈伯斯在史丹佛大學畢業典禮的演講上所說的：

> 生命是一場連點點的遊戲，規則是你只能往過去連接，不能往未來連接。

> 往過去看相當清楚，但往未來看卻一片模糊 ...

不過，有些程式人試圖創造出《可以融合 LLM 的程式設計的框架》，像是：

1. https://github.com/langchain-ai/langchain
2. https://github.com/stanfordnlp/dspy

不管是用 Prompt ，或者結合 System2 ，在 AI 取代人類之前，我們還有得玩 ...

最後、讓我們問一個問題：

在 LLM 技術被發展出來之後，我們該怎麼樣有效使用 LLM 技術來建構更厲害的系統呢？

這個問題有兩種不同的結局，分別是：

1. LLM 自己寫程式插到自己身上，然後建構出更厲害的系統
2. 程式人寫程式去呼叫 LLM ，然後建構出更厲害的系統

在 1 還沒被實現之前，程式人都還有得混 ...

而且、你可以用 LLM 來寫程式，然後再插回 LLM 身上 ...



