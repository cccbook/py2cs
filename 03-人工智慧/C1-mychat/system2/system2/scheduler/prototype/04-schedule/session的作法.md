
* https://stackoverflow.com/questions/74711107/openai-api-continuing-conversation-in-a-dialogue

似乎只能將傳入的 messages 陣列不斷延長 ...

```py
import os
import openai


class ChatApp:
    def __init__(self):
        # Setting the API key to use the OpenAI API
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.messages = [
            {"role": "system", "content": "You are a coding tutor bot to help user write and optimize python code."},
        ]

    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        return response["choices"][0]["message"]
```
