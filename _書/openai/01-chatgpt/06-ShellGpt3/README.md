# ShortGPT

```
$ shortgpt.sh
Welcome to shortgpt. You may use the following commands
1. quit
2. history
3. shell <command>
4. chat <prompt>
5. fchat <file> <prompt>

You may use the following $key for short
{
  "mt": "翻譯下列文章",
  "tw": "以 繁體中文 格式輸出",
  "en": "output in English",
  "jp": "output in Japanese",
  "md": "format in Markdown+LaTex, add space before and after $..$"
}

command> chat 你好
========question=======
你好
========response=======
你好！我是一個聊天機器人，我可以回答你的問題和進行對話。有什麼我能為你做的嗎？

command> chat 2018 年的美國總統是誰?
========question=======
2018 年的美國總統是誰?
========response=======
2018年的美國總統仍然是唐納德·特朗普。

command> fchat GPT.md GPT 是甚麼? $tw
========question=======
GPT 是甚麼? 以 繁體中文 格式輸出
========response=======
Response will write to file:GPT.md

command> shell ls
GPT.md

command> shell cat GPT.md
GPT 是一款由 OpenAI 所開發的自然語言生成模型，英文全名為 "Generative Pre-trained
Transformer"，翻譯成中文為「生成式預訓練轉換器」。它可以生成自然的文本，例如文章
、網路評論、電子郵件等，並且能夠用戶進行對話。
command> chat $mt $en file:GPT.md
========question=======
翻譯下列文章 output in English

GPT 是一款由 OpenAI 所開發的自然語言生成模型，英文全名為 "Generative Pre-trained
Transformer"，翻譯成中文為「生成式預訓練轉換器」。它可以生成自然的文本，例如文章
、網路評論、電子郵件等，並且能夠用戶進行對話。
========response=======
The article is about GPT, a natural language generation model developed by OpenAI. Its full name is "Generative Pre-trained Transformer" and it can generate natural text such as articles, online comments, emails, and participate in user conversations.

command> history
0:chat 你好
1:chat 2018 年的美國總統是誰?
2:fchat GPT.md GPT 是甚麼? $tw
3:shell ls
4:shell cat GPT.md
5:chat $mt $en file:GPT.md
6:history

command> quit
```
