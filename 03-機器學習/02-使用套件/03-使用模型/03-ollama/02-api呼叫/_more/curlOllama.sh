curl -X POST http://localhost:11434/api/generate \
  -d '{
    "model": "mistral",
    "prompt": "Explain quantum computing in simple terms.",
    "stream": false
  }' \
  -H "Content-Type: application/json"
