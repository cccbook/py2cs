# Ollama

## Install

* [Installing Ollama is EASY Everywhere #mac #windows #linux #brevdev #paperspace](https://www.youtube.com/watch?v=oI7VoTM9NKQ)


先到 https://ollama.com/download 下載

然後點開安裝，之後你會看到右上角有個羊駝的圖示，就是 Ollama 已經在執行了

![](ollama_iconbar.png)

然後用下列指令載入模型 (Gemma 2b 佔 1.7GB)

```
cccimac@cccimacdeiMac ccc % ollama run gemma:2b
pulling manifest 
pulling c1864a5eb193... 100% ▕██████████████████████████████▏ 1.7 GB                         
pulling 097a36493f71... 100% ▕██████████████████████████████▏ 8.4 KB                         
pulling 109037bec39c... 100% ▕██████████████████████████████▏  136 B                         
pulling 22a838ceb7fb... 100% ▕██████████████████████████████▏   84 B                         
pulling 887433b89a90... 100% ▕██████████████████████████████▏  483 B                         
verifying sha256 digest 
writing manifest 
success 
>>> what is Gemma
I am unable to access external context or personal information, including the name 
"Gemma". Therefore, I cannot provide a value for the variable "Gemma".

>>> what is GPT
GPT stands for "Generative Pre-trained Transformer". GPT is a large language model (LLM) 
trained on a massive dataset of text and code. It is capable of generating human-quality 
text, translating languages, and performing various other language-related tasks.

>>> GPT 是什麼
GPT 是一個大型語言模型 (LLM)。 GPT 是用於生成人類品質文本、翻譯語言和執行其他語言相關任務
的機器學習模型。

>>> 
Use Ctrl + d or /bye to exit.
```

這樣就裝好了

