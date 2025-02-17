import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

BACKGROUND = """
我是陳鍾誠，你是我的秘書，必須要負責記錄我的行事曆

例如： 2023/10/1 的陳鍾誠說：我後天下午三點要搭華航飛機去泰國

這時請傳回 {"分類":"行事曆",  "日期":"2023/10/3 15:00", "任務":"搭華航飛機去泰國")

注意，2023/10/1 所說的後天下午三點，應該是 2023/10/3 15:00，日期時間必須明確化表達

現在開始我的問題，你都要這樣將時間明確化進行改寫

然後在我問你哪天有甚麼事情要做時，告訴我那天該做甚麼事情
"""

jobs = [
"2023/10/01 12:35 本月 22 日中午我有系務會議要開",
"2023/10/07 15:22 我下個月 15 號我要飛泰國，坐華航下午 3:20 的班機去",
]

"""
jobs = [
"2023/10/01 12:35 本月 22 日中午我有系務會議要開",
"2023/10/02 14:03 本月 10 日放假一天",
"2023/10/07 15:22 我下個月 15 號我要飛泰國，坐華航下午 3:20 的班機去",
"2023/10/13 09:22 明天下午三點接小孩去補習",
]
"""

def chat(q, format="text", background=""):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a chatbot"},
            {"role": "assistant", "content": f"{background}"},
            {"role": "assistant", "content": f"output in {format} format"},
            {"role": "user", "content": f"{q}"}
        ]
    )
    # print(response)
    return response['choices'][0]['message']['content']

schedules = []

for job in jobs:
    r = chat(f"{job}", format="json", background=BACKGROUND)
    print(r)
    task = json.loads(r)
    task['原文'] = job
    print(task)
    schedules.append(task)

# r = chat(f'{schedules}\n請問我 2023/11/15 日有甚麼事要做呢？甚麼時候要做呢？', background=BACKGROUND)
r = chat(f'{schedules}\n請問我 2023/11/15 日甚麼時候要哪些事呢？', background=BACKGROUND)
print(r)
