curl -X POST http://localhost:11434/api/chat \
     -d '{ "model": "llama3.2:3b", "messages": [{"role":"user", "content":"什麼是群論？"}], 
     "stream":false }' \
     -H "Content-Type: application/json"
