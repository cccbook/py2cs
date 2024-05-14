import os
import sys
from openai import OpenAI

openai_key = os.environ.get("OPENAI_API_KEY")
openai = OpenAI() if openai_key else None

def openaiChat(question):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "請用繁體中文 zh-tw 回答"},
            {"role": "assistant", "content": "你聊天對象是台灣人"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

from groq import Groq

groq_key = os.environ.get("GROQ_API_KEY")
groq = Groq(api_key=groq_key) if groq_key else None

def groqChat(question):
    response = groq.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "請用繁體中文 zh-tw 回答"},
            {"role": "assistant", "content": "你聊天對象是台灣人"},
            {"role": "user", "content": question},
        ],
        model="llama3-8b-8192",
    )
    return response.choices[0].message.content

def chatList():
    rlist = []
    if groq: rlist.append('groq')
    if openai: rlist.append('openai')
    return rlist

def chat(question, model="groq"):
    print('model=', model)
    if model=='groq':
        return groqChat(question)
    elif model=='openai':
        return openaiChat(question)
    else:
        raise Exception('chat model not found!')

if __name__ == '__main__':
    clist = chatList()
    print('chatList = ', clist)
    tmodel = sys.argv[1] if sys.argv[1] in clist else clist[0]
    print('tmodel=', tmodel)
    response = chat(" ".join(sys.argv[1:]), model=tmodel)
    print(response)

