# 


* [Best practices for prompt engineering with OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)
* [OPENAI:API REFERENCE](https://platform.openai.com/docs/api-reference/introduction)
* [ChatGPT API Transition Guide](https://help.openai.com/en/articles/7042661-chatgpt-api-transition-guide)


* [ChatGPT来了，让我们快速做个AI应用](https://juejin.cn/post/7215424831209750589), 作者：极客时间App

在OpenAI的官方文档里，可以看到这个接口也非常简单。你需要传入的参数，从一段Prompt变成了一个数组，数组的每个元素都有role和content两个字段。

role这个字段一共有三个角色可以选择，分别是 system 代表系统，user代表用户，而assistant则代表AI的回答。
当role是system的时候，content里面的内容代表我们给AI的一个指令，也就是告诉AI应该怎么回答用户的问题。比如我们希望AI都通过中文回答，我们就可以在content里面写“你是一个只会用中文回答问题的助理”，这样即使用户问的问题都是英文的，AI的回复也都会是中文的。

而当role是user或者assistant的时候，content里面的内容就代表用户和AI对话的内容。和我们在第03讲里做的聊天机器人一样，你需要把历史上的对话一起发送给OpenAI的接口，它才有能够理解整个对话的上下文的能力。


```py
import openai
openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)
```


* [【技術分享】2023/03/01 ChatGPT API可以用啦！快速Setup你的ChatGPTAPI](https://axk51013.medium.com/%E6%8A%80%E8%A1%93%E5%88%86%E4%BA%AB-2023-03-01-chatgpt-api%E5%8F%AF%E4%BB%A5%E7%94%A8%E5%95%A6-2435b6d23bbd)

role有三種：”system”、"user"、”assistant”，一般剛開始使用chatGPT的時候會先給一個system message來教chatGPT如何表現，不過gpt-3.5-turbo模型目前並沒有很明顯在理system message，如果希望chatGPT如何表現，還是後續進行prompt engineering比較好（之後會說明）

"user"：是輸入作為用戶你想要講的話、問的問題

"assistant"：是前幾個round的ChatGPT回復，所以如果要做multi-round的chatGPT，必須重複的紀錄回答的歷史並重複輸入（可能有更好方法）。