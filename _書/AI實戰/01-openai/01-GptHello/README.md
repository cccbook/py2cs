# 呼叫 OpenAI

參考

1. https://platform.openai.com/docs/libraries
2. [只要會寫 Python 的人，都能簡單上手的 ChatGPT Python API](https://pecutsai.medium.com/%E5%8F%AA%E8%A6%81%E6%9C%83%E5%AF%AB-python-%E7%9A%84%E4%BA%BA-%E9%83%BD%E8%83%BD%E7%B0%A1%E5%96%AE%E4%B8%8A%E6%89%8B%E7%9A%84-chatgpt-python-api-cc4d3ed2234a)

## 用法

1. 登入 OpenAI
2. 選右上角你帳戶功能表中的 View API key
3. 按 Create a new secret key
4. 把 key 設到 OPENAI_API_KEY 這個環境變數中
5. 執行 GptHello.py 程式

## 執行過程

```
$ pip install openai
...
$ echo $OPENAI_API_KEY
...
```

準備好之後，可以先用 openai 的命令工具先測試

```
ccckmit@asus MINGW64 /d/ccc/py2cs/_書/AI實戰/01-openai/01-ChatGPT1 (master)
$ openai api completions.create -m text-davinci-003 -p
"Say this is a test" -t 0 -M 7 --stream
Say this is a test

This is indeed a test
ccckmit@asus MINGW64 /d/ccc/py2cs/_書/AI實戰/01-openai/01-ChatGPT1 (master)
$ openai api completions.create -m text-davinci-003 -p
"GPT 是甚麼?" -t 0 -M 7 --stream
GPT 是甚麼?

GPT 是
ccckmit@asus MINGW64 /d/ccc/py2cs/_書/AI實戰/01-openai/01-ChatGPT1 (master)
$ openai api completions.create -m text-davinci-003 -p
"GPT 是甚麼?" -t 0 -M 300 --stream
GPT 是甚麼?

GPT 是一種自然語言生成技術，它可以根據輸入的文本，自動生
成新的文本。它是一種基於機器學習的技術，可以讓機器學習如
何根據輸入的文本來生成新的文本。GPT 可以用於自然語言理解
、自然語言生成、文本分析和自動文檔生成等應用。
```

然後就可以跑 GptHello.py 了

```
$ python GptHello.py
{
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "logprobs": null,
      "text": "\n\nThis is indeed a test"
    }
  ],
  "created": 1679872443,
  "id": "cmpl-6yTqln3eZK6fdBG2LQCoOU1f4yzB0",
  "model": "text-davinci-003",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 7,
    "prompt_tokens": 5,
    "total_tokens": 12
  }
}

$ python GptHello2.py
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": "\n\nGPT \u662f\u4e00\u7a2e\u81ea\u7136\u8a9e\u8a00\u751f\u6210\u6280\u8853\uff0c\u5b83\u53ef\u4ee5\u6839\u64da\u8f38\u5165\u7684\u6587\u672c\uff0c\u81ea\u52d5\u751f\u6210\u65b0\u7684\u6587\u672c\u3002\u5b83\u662f\u4e00\u7a2e\u57fa\u65bc\u6a5f\u5668\u5b78\u7fd2\u7684\u6280\u8853\uff0c\u53ef\u4ee5\u8b93\u6a5f\u5668\u5b78\u7fd2\u5982\u4f55\u6839\u64da\u8f38\u5165\u7684\u6587\u672c\u4f86\u751f\u6210\u65b0\u7684\u6587\u672c\u3002GPT \u53ef\u4ee5\u7528\u65bc\u81ea\u7136\u8a9e\u8a00\u7406\u89e3\u3001\u81ea\u7136\u8a9e\u8a00\u751f\u6210\u3001\u6587\u672c\u5206\u6790\u548c\u81ea\u52d5\u6587\u6a94\u751f\u6210\u7b49\u61c9\u7528\u3002"
    }
  ],
  "created": 1679877714,
  "id": "cmpl-6yVDmq7cYFjs9GnL0Qvim4OF70R6o",
  "model": "text-davinci-003",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 226,
    "prompt_tokens": 10,
    "total_tokens": 236
  }
}

ccckmit@asus MINGW64 /d/ccc/py2cs/_書/AI實戰/01-openai/01-ChatGPT1 (master)
$ python GptHello3.py


GPT 是一種自然語言生成技術，它可以根據輸入的文本，自動生
成新的文本。它是一種基於機器學習的技術，可以讓機器學習如
何根據輸入的文本來生成新的文本。GPT 可以用於自然語言理解
、自然語言生成、文本分析和自動文檔生成等應用。
```

