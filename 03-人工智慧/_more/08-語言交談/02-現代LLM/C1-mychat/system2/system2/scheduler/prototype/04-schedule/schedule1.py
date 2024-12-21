import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

MEMORY = """
# 陳鍾誠的行事紀錄

陳鍾誠:2023/10/01 12:35 本月 22 日中午我有系務會議要開
陳鍾誠:2023/10/02 14:03 本月 10 日放假一天
陳鍾誠:2023/10/07 15:22 我 11 月 15 號我要飛泰國，坐華航下午 3:20 的班機去
陳鍾誠:2023/10/13 09:22 明天下午三點接小孩去補習
"""

def chat(q):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a chatbot"},
            {"role": "user", "content": f"{MEMORY}\n\n{q}"}
        ]
    )
    print('memory=', MEMORY)
    print(response)
    return response['choices'][0]['message']['content']

r = chat('請問陳鍾誠 2023/11/15 日有甚麼事要做呢？')
print(r)
