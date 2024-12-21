# 架設網站呼叫 imac 上的 ollama 服務

從 [chatGpt.md](chatGPT.md) 對話中，我發現使用 ssh 反向代理應該會比較適合我的案例

方法如下


### 方法 1：使用 **SSH 反向代理（SSH Reverse Proxy）**
這是一個相對簡單且安全的方式，利用您已經有公共 IP 的雲端主機作為跳板，通過 SSH 建立一個反向代理，將雲端主機的某個端口映射到 iMac 的本地 Ollama 服務。

#### 步驟：

1. **在 iMac 上啟動 Ollama 服務**：
    - 確保 iMac 上的 Ollama 服務正在運行。例如，服務可能在 `localhost:11434` 上。

2. **設置 SSH 反向代理**：
    - 在 iMac 上，使用以下命令通過 SSH 與您的雲端主機建立連接，並將 iMac 上的 `11434` 端口映射到雲端主機的某個端口（例如 `8080`）：

    ```bash
    ssh -R 8080:localhost:11434 your_username@your_cloud_server_ip
    ```

    - 解釋：
        - `-R 8080:localhost:11434` 表示將雲端主機的 `8080` 端口映射到 iMac 上 `localhost:11434` 的 Ollama 服務。
        - `your_username@your_cloud_server_ip` 是您在雲端主機上的 SSH 用戶名和 IP。

3. **在雲端主機上訪問 Ollama 服務**：
    - 現在，您可以在雲端主機上通過 `http://localhost:8080/api/chat` 訪問 iMac 上的 Ollama 服務。
    - 您的雲端主機會將所有對 `localhost:8080` 的請求轉發到 iMac 上的 `localhost:11434`。

#### 注意事項：
- iMac 需要能夠連接到雲端主機（SSH 必須能連通）。
- 您可以使用 `autossh` 等工具保持這個連接持續運行，即使 SSH 連線中斷也能自動重連。


## 測試結果

我用 8080 去映射，然後用 curl 測


```
root@localhost:~# curl http://localhost:8080/api/chat -d '{
  "model": "gemma:2b",
  "messages": [
    { "role": "user", "content": "你是誰？" }                       
      
  ]
}'

{"model":"gemma:2b","created_at":"2024-09-29T03:26:35.995306Z","message":{"role":"assistant","content":"我"},"done":false}
{"model":"gemma:2b","created_at":"2024-09-29T03:26:36.016678Z","message":{"role":"assistant","content":"是一個"},"done":false}
{"model":"gemma:2b","created_at":"2024-09-29T03:26:36.038308Z","message":{"role":"assistant","content":" AI"},"done":false}
{"model":"gemma:2b","created_at":"2024-09-29T03:26:36.059308Z","message":{"role":"assistant","content":" 人"},"done":false}
{"model":"gemma:2b","created_at":"2024-09-29T03:26:36.079925Z","message":{"role":"assistant","content":"工"},"done":false}
{"model":"gemma:2b","created_at":"2024-09-29T03:26:36.100878Z","message":{"role":"assistant","content":"智慧"},"done":false}
{"model":"gemma:2b","created_at":"2024-09-29T03:26:36.121738Z","message":{"role":"assistant","content":"，"},"done":false}
{"model":"gemma:2b","created_at":"2024-09-29T03:26:36.143032Z","message":{"role":"assistant","content":"來自"},"done":false}
{"model":"gemma:2b","created_at":"2024-09-29T03:26:36.163716Z","message":{"role":"assistant","content":" Google"},"done":false}
{"model":"gemma:2b","created_at":"2024-09-29T03:26:36.184575Z","message":{"role":"assistant","content":"。"},"done":false}
{"model":"gemma:2b","created_at":"2024-09-29T03:26:36.205398Z","message":{"role":"assistant","content":"\n\n"},"done":false}
{"model":"gemma:2b","created_at":"2024-09-29T03:26:36.352379Z","message":{"role":"assistant","content":""},"done_reason":"stop","done":true,"total_duration":4554453250,"load_duration":4118380125,"prompt_eval_count":30,"prompt_eval_duration":77384000,"eval_count":18,"eval_duration":357169000}
```

## 測試 2

我用 11434 去映射，然後用 ollamaApi.py 測

```
(base) cccimac@cccimacdeiMac 03-callImac % ./reverseOllama.sh 
root@139.162.90.34's password: 
Welcome to Ubuntu 24.04.1 LTS (GNU/Linux 6.8.0-41-generic x86_64)

root@localhost:~# git clone git@github.com:ccc-py/fastai.git
Cloning into 'fastai'...
remote: Enumerating objects: 53, done.
remote: Counting objects: 100% (53/53), done.
remote: Compressing objects: 100% (37/37), done.
remote: Total 53 (delta 13), reused 47 (delta 10), pack-reused 0 (from 0)
Receiving objects: 100% (53/53), 298.22 KiB | 433.00 KiB/s, done.
Resolving deltas: 100% (13/13), done.
root@localhost:~# ls
fastai
root@localhost:~# cd fastai
root@localhost:~/fastai# ls
doc  LICENSE  README.md  test
root@localhost:~/fastai# cd test
root@localhost:~/fastai/test# ls
01-ollamaApi  02-fastapi1  03-callImac  A1-websocket
root@localhost:~/fastai/test# cd 03-callImac/
root@localhost:~/fastai/test/03-callImac# ls
chatGpt.md  ollamaApi.py  README.md  reverseOllama.sh  test.md
root@localhost:~/fastai/test/03-callImac# ls
chatGpt.md  ollamaApi.py  README.md  reverseOllama.sh  test.md
root@localhost:~/fastai/test/03-callImac# python ollamaApi.py
Command 'python' not found, did you mean:
  command 'python3' from deb python3
  command 'python' from deb python-is-python3
root@localhost:~/fastai/test/03-callImac# python3
Python 3.12.3 (main, Sep 11 2024, 14:17:37) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> quit()
root@localhost:~/fastai/test/03-callImac# ls
chatGpt.md  ollamaApi.py  README.md  reverseOllama.sh  test.md
root@localhost:~/fastai/test/03-callImac# python3 ollamaApi.py
我是一個 AI 語言模型，專門在與您互動。

我是一個程式設計的語言模型，可以理解和回應您的自然語言指令。root@localhost:~/fastai/test/03-callImac# 
```

