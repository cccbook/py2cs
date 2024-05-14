from openai import OpenAI



client = OpenAI(
    # api_key="My API Key" # defaults to os.environ.get("OPENAI_API_KEY")
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "請問 321+123 是多少?",
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion)
print(chat_completion.choices[0].message.content)
