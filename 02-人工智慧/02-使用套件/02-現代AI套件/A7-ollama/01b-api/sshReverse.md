# ssh reverse proxy

## 指令

```
ssh -R 11434:localhost:11434 root@139.162.90.34
```

## 測試

說明：不能在其他電腦直接 curl http://139.162.90.34:11434/api/chat，會失敗

但是可以在 139.162.90.34 上執行下列指令

推測：我只要在 139.162.90.34 上架個 deno server 就可以提供對外服務了。

```sh
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [
    { "role": "user", "content": "你是誰？" }                 
  ]
}'
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.138506Z","message":{"role":"assistant","content":"我"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.174593Z","message":{"role":"assistant","content":"是"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.20684Z","message":{"role":"assistant","content":"G"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.239626Z","message":{"role":"assistant","content":"PT"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.275603Z","message":{"role":"assistant","content":"-"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.311575Z","message":{"role":"assistant","content":"3"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.346261Z","message":{"role":"assistant","content":"，"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.380797Z","message":{"role":"assistant","content":"一个"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.41535Z","message":{"role":"assistant","content":"高"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.449936Z","message":{"role":"assistant","content":"级"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.485159Z","message":{"role":"assistant","content":" chat"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.519802Z","message":{"role":"assistant","content":"bot"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.554522Z","message":{"role":"assistant","content":"。"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.589093Z","message":{"role":"assistant","content":"我的"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.623912Z","message":{"role":"assistant","content":"目"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.658742Z","message":{"role":"assistant","content":"的是"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.726636Z","message":{"role":"assistant","content":"幫"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.761427Z","message":{"role":"assistant","content":"助"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.794433Z","message":{"role":"assistant","content":"您"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.829159Z","message":{"role":"assistant","content":"尋"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.863812Z","message":{"role":"assistant","content":"找"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.89859Z","message":{"role":"assistant","content":"資"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.932824Z","message":{"role":"assistant","content":"訊"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:34.968095Z","message":{"role":"assistant","content":"、"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:35.00351Z","message":{"role":"assistant","content":"提供"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:35.038325Z","message":{"role":"assistant","content":"คำ"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:35.073217Z","message":{"role":"assistant","content":"提"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:35.107867Z","message":{"role":"assistant","content":"醒"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:35.143935Z","message":{"role":"assistant","content":"和"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:35.178167Z","message":{"role":"assistant","content":"回答"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:35.212799Z","message":{"role":"assistant","content":"您的"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:35.247811Z","message":{"role":"assistant","content":"問題"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:35.282451Z","message":{"role":"assistant","content":"。"},"done":false}
{"model":"llama3.2:3b","created_at":"2024-10-18T04:05:35.317056Z","message":{"role":"assistant","content":""},"done_reason":"stop","done":true,"total_duration":1329316125,"load_duration":33156584,"prompt_eval_count":29,"prompt_eval_duration":116628000,"eval_count":35,"eval_duration":1178533000}
```