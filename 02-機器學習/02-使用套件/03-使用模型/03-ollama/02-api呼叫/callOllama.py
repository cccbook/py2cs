import requests

response = requests.post(
    'http://localhost:11434/api/chat',
    json={
        'model': 'llama3.2:3b',
        'messages': [
            {
                'role': 'user',
                'content': '什麼是群論？'
            }
        ],
        'stream':False,
    }
)

# 這個端點返回單一 JSON 物件
text_only = response.json()['message']['content']
print(text_only)
