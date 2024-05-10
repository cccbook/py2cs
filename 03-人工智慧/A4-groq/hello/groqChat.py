import os
import sys
from groq import Groq

question = " ".join(sys.argv[1:])
print("問題：", question)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": question,
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)
