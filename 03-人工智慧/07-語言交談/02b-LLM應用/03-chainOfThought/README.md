# COT

對於 ChatGPT 這類的 AI LLM

你只要在問題後面加上

Let's do it step by step (請一步一步推導)

經常就可以讓原本答錯的問題，變成正確的

這個技巧稱為 COT (Chain Of Thought)

## Zero-Shot-CoT

基本上就是加上一條： 

> Let's think step by step ... (請一步一步思考並寫下思考過程)


## 範例

You
我媽媽的弟弟的兒子，親屬稱謂是甚麼？

Copilot
您媽媽的弟弟的兒子，親屬稱謂是舅舅1。如果您還有其他關於親戚稱謂的問題，歡迎隨時詢問！


You
我媽媽的弟弟的兒子，親屬稱謂是甚麼？

do it step by step

Copilot
你的媽媽的弟弟就是你的叔叔，而你叔叔的兒子則是你的表弟。所以，你和你叔叔的兒子之間的親屬稱謂是 表弟。如果你還有其他問題，歡迎繼續詢問！

## 思考

為何 COT 可以讓 LLM 變聰明呢？ (想想《媽媽的弟弟的兒子》的例子)

## 後續實驗

User
我媽媽的弟弟的兒子，親屬稱謂是甚麼？

ChatGPT
你媽媽的弟弟的兒子就是你的表弟。

User
我媽媽的弟弟的兒子的女兒，親屬稱謂是甚麼？

ChatGPT
你媽媽的弟弟的兒子的女兒是你的表侄女。

User
我媽媽的弟弟的兒子的女兒的爸爸，親屬稱謂是甚麼？

ChatGPT
這個人是你的表兄弟。


## 參考文獻

* https://learnprompting.org/zh-tw/docs/intermediate/chain_of_thought

* https://www.promptingguide.ai/techniques/cot


* [大模型思维链（Chain-of-Thought）技术原理](https://www.zhihu.com/tardis/zm/art/629087587?source_id=1003) (讚)
* [Chain of Thought 开山之作论文详解](https://zhuanlan.zhihu.com/p/582758381)

## Zero-Shot-CoT

基本上就是加上一條： 

> Let's think step by step ... (請一步一步思考並寫下思考過程)

## Manual-CoT

在输入问题之前，手动设计一些问题和答案的样例（样例的答案给出中间推理步骤），这些问题和答案都需要手动构造，所以叫 Manual-CoT

语言模型的输入是一些手动设计的问题和答案的参考样例连接一个真正需要求解的问题，然后让语言模型进行续写

Manual-CoT 比 Zero-Shot-CoT 的性能要好，因为它采用的是 few shot ，在输入中提供了一些问题、中间推理步骤以及答案的样例给语言模型进行参考。但是，提供这些样例需要进行人工设计，这就需要一定的人工成本

## Auto-CoT

通过多样性选取有代表性的问题

对于每一个采样的问题拼接上“Let's think step by step”（类似于 Zero-Shot-CoT ）输入到语言模型，让语言模型生成中间推理步骤和答案，然后把这些所有采样的问题以及语言模型生成的中间推理步骤和答案全部拼接在一起，构成少样本学习的样例，最后再拼接上需要求解的问题一起输入到语言模型中进行续写，最终模型续写出了中间的推理步骤以及答案
