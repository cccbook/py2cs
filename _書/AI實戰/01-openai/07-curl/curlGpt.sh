# 注意，中文會失敗，例如 "GPT 是甚麼?"
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "What is GPT?"}],
     "temperature": 0.7
   }'