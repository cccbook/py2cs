export async function chat(question) {
    let r = await fetch("https://api.openai.com/v1/chat/completions", { 
        body: JSON.stringify({
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.7
        }),
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${Deno.env.get('OPENAI_API_KEY')}`,
        }
    })
    let json = await r.json()
    // console.log('json=', json)
    return {answer:json.choices[0].message.content}
}
