from openai import OpenAI

client = OpenAI(
    # api_key="My API Key" # defaults to os.environ.get("OPENAI_API_KEY")
)

# 關於 role: 1. user (使用者) 2. system (背景知識) 3. assistant (AI) 的說明請看
# https://ai.stackexchange.com/questions/39837/meaning-of-roles-in-the-api-of-gpt-4-chatgpt-system-user-assistant
chat_completion = client.chat.completions.create(
    messages=[
        { "role": "system", "content": "你是程式專家" },
        { "role": "user", "content": "請問 321+123 是多少?" },
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion)
print(chat_completion.choices[0].message.content)