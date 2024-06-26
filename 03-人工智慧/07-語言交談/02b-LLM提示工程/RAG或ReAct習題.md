https://github.com/ccc112b/ai/issues/12

參考

1. [提示工程](https://github.com/cccbook/py2gpt/wiki/prompt)
2. https://www.langchain.com/
3. https://console.groq.com/docs/quickstart
4. [02b-LLM提示工程](https://github.com/ccc112b/py2cs/tree/master/03-%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7/07-%E8%AA%9E%E8%A8%80%E4%BA%A4%E8%AB%87/02b-LLM%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B)
5. https://github.com/stanfordnlp/dspy
    * [dspy.GROQ](https://dspy-docs.vercel.app/api/language_model_clients/Groq)
6. RAG 可參考 -- https://www.perplexity.ai/
    * https://www.promptingguide.ai/zh/techniques/rag
    * https://www.google.com/search?q=langchain+rag
    * [RAG實作教學，LangChain + Llama2 |創造你的個人LLM](https://medium.com/@cch.chichieh/rag%E5%AF%A6%E4%BD%9C%E6%95%99%E5%AD%B8-langchain-llama2-%E5%89%B5%E9%80%A0%E4%BD%A0%E7%9A%84%E5%80%8B%E4%BA%BAllm-d6838febf8c4)
    * LangChain: [Use cases](https://python.langchain.com/v0.1/docs/use_cases/) / [Q&A with RAG](https://python.langchain.com/v0.1/docs/use_cases/question_answering/)
    * DSPy: [RAG](https://dspy-docs.vercel.app/docs/tutorials/rag)
7. ReAct 可參考
    * https://www.promptingguide.ai/zh/techniques/react
    * https://www.google.com/search?q=langchain+react
    * langchain: [ReAct](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/)
    * DSPy: [ReAct](https://dspy-docs.vercel.app/docs/deep-dive/modules/react)
    * [讓 LLM 更好用的方法：ReAct prompting](https://edge.aif.tw/application-react-prompting/)

ReAct 的一種可能提示詞

```py
sys_prompt = """
You run in a loop of Thought, Action, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you。
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

call_google:
e.g. call_google: European Union
Returns a summary from searching European Union on google

You can look things up on Google if you have the opportunity to do so, or you are not sure about the query

Example1:

Question: What is the capital of Australia?
Thought: I can look up Australia on Google
Action: call_google: Australia

You will be called again with this:

Observation: Australia is a country. The capital is Canberra.

You then output:

Answer: The capital of Australia is Canberra

Example2:

Question: What are the constituent countries of the European Union? Try to list all of them.
Thought: I can search the constituent countries in European Union by Google
Action: call_google: constituent countries in European Union

Observation: The European Union is a group of 27 countries in Europe, Today, 27 countries are part of the European Union.
These countries are:

Austria、Belgium、Bulgaria、Croatia、Cyprus、Czechia、Denmark、Estonia、Finland、France、Germany、Greece、Hungary、Ireland、
Italy、Latvia、Lithuania、Luxembourg、Malta、Netherands、Poland、Portugal、Romania、Slovakia、Slovenia、Spain、Sweden

Answer: There are 27 constituent countries in European Union, the name of constituent countries are listed as follows
Austria
Belgium
Bulgaria
Croatia
Cyprus
Czechia
Denmark
Estonia
Finland
France
Germany
Greece
Hungary
Ireland
Italy
Latvia
Lithuania
Luxembourg
Malta
Netherlands
Poland
Portugal
Romania
Slovakia
Slovenia
Spain
Sweden

""".strip()
```
