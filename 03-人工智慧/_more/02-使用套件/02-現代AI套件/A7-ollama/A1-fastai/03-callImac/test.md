
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
