import requests

url = "http://localhost:11434/api/generate"
data = {
    "model": "mistral",
    "prompt": "What is the capital of France?",
    "stream": False
}

response = requests.post(url, json=data)
print(response.json()["response"])  # 會輸出 "Paris"
