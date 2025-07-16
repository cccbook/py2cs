curl -X POST http://localhost:11434/api/generate \
     -d '{ "model": "llama3.2:3b", "prompt": "什麼是群論？", "stream":false }' \
     -H "Content-Type: application/json"
