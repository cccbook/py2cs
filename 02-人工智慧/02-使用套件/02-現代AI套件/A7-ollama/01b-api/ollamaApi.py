import requests
import json

# url = 'http://localhost:11434/api/chat'
url = 'http://139.162.90.34::11434/api/chat'
data = {
    "model": "llama3.2:3b", #"gemma:2b",
    "messages": [
        {"role": "user", "content": "你是誰？"}
    ]
}

response = requests.post(url, json=data)
# print(response.text) # 格式不對，不能直接用 json # print(response.json())
lines = response.text.strip().split('\n')
json1 = '['+',\n'.join(lines)+']'
# print(json1)
response = json.loads(json1)
print(json.dumps(response, indent=2, ensure_ascii=False))

for t in response:
    print(t['message']['content'], end='')