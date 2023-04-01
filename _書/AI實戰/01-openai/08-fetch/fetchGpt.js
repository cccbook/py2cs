const jsonResponse = await fetch("https://api.openai.com/v1/chat/completions", 
{
    body: JSON.stringify({
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "GPT 是甚麼?"}],
        "temperature": 0.7
    }),
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${Deno.env.get('OPENAI_API_KEY')}`,
    }
})
const jsonData = await jsonResponse.json()
console.log(JSON.stringify(jsonData, null, 2))
