from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import requests
import json

app = FastAPI()

# 提供靜態文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 定義 Ollama API 的相關設定
OLLAMA_API_URL = 'http://localhost:11434/api/chat'
OLLAMA_MODEL = 'gemma:2b'

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()  # 接收來自客戶端的訊息
            print(f"Received message from client: {data}")

            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "user", "content": data}
                ]
            }

            try:
                response = requests.post(OLLAMA_API_URL, json=payload)  # 發送 POST 請求
                response.raise_for_status()
            except requests.RequestException as e:
                await websocket.send_text(f"Error contacting Ollama API: {e}")
                continue

            lines = response.text.strip().split('\n')
            json_str = '[' + ',\n'.join(lines) + ']'

            try:
                response_json = json.loads(json_str)
            except json.JSONDecodeError as e:
                await websocket.send_text(f"Error parsing Ollama API response: {e}")
                continue

            answer = ''.join([item['message']['content'] for item in response_json])
            await websocket.send_text(answer)

    except WebSocketDisconnect:
        print("Client disconnected")

# 如果需要運行此腳本，請取消以下註解並執行
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
