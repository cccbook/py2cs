

```
root@localhost:~# curl -fsSL https://ollama.com/install.sh | sh
>>> Installing ollama to /usr/local
>>> Downloading Linux amd64 bundle
######################################################################## 100.0%
>>> Creating ollama user...
>>> Adding ollama user to render group...
>>> Adding ollama user to video group...
>>> Adding current user to ollama group...
>>> Creating ollama systemd service...
>>> Enabling and starting ollama service...
Created symlink /etc/systemd/system/default.target.wants/ollama.service → /etc/systemd/system/ollama.service.
>>> The Ollama API is now available at 127.0.0.1:11434.
>>> Install complete. Run "ollama" from the command line.
WARNING: No NVIDIA/AMD GPU detected. Ollama will run in CPU-only mode.
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