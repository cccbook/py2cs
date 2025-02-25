from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import requests
import json

app = FastAPI()

# 定義 Ollama API 的相關設定
OLLAMA_API_URL = 'http://localhost:11434/api/chat'
OLLAMA_MODEL = 'gemma:2b'

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 接收來自客戶端的訊息
            data = await websocket.receive_text()
            print(f"Received message from client: {data}")

            # 準備發送到 Ollama API 的資料
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "user", "content": data}
                ]
            }

            try:
                # 發送 POST 請求到 Ollama API
                response = requests.post(OLLAMA_API_URL, json=payload)
                response.raise_for_status()  # 確認請求成功
            except requests.RequestException as e:
                error_message = f"Error contacting Ollama API: {e}"
                print(error_message)
                await websocket.send_text(error_message)
                continue

            # 處理回應
            lines = response.text.strip().split('\n')
            json_str = '[' + ',\n'.join(lines) + ']'
            try:
                response_json = json.loads(json_str)
            except json.JSONDecodeError as e:
                error_message = f"Error parsing Ollama API response: {e}"
                print(error_message)
                await websocket.send_text(error_message)
                continue

            # 收集回應內容
            answer = ''.join([item['message']['content'] for item in response_json])
            print(f"Sending answer to client: {answer}")

            # 將答案發送回客戶端
            await websocket.send_text(answer)

    except WebSocketDisconnect:
        print("Client disconnected")

# 如果需要運行此腳本，請取消以下註解並執行
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
