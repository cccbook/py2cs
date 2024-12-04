# 架設 ollama server 

在我的 imac 上沒問題，架設成功

但在 linode $5 美元主機上，連 gemma:2b 都因記憶體不足，無法執行

```
root@localhost:~# ollama run gemma:2b
pulling manifest 
pulling c1864a5eb193... 100% ▕██████████████████████████████▏ 1.7 GB                         
pulling 097a36493f71... 100% ▕██████████████████████████████▏ 8.4 KB                         
pulling 109037bec39c... 100% ▕██████████████████████████████▏  136 B                         
pulling 22a838ceb7fb... 100% ▕██████████████████████████████▏   84 B                         
pulling 887433b89a90... 100% ▕██████████████████████████████▏  483 B                         
verifying sha256 digest 
writing manifest 
success 
Error: model requires more system memory (1.7 GiB) than is available (1.2 GiB)
```