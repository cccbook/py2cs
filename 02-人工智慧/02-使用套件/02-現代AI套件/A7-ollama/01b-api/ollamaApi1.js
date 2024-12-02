// import fetch from "node-fetch";

// const url = 'http://localhost:11434/api/chat';
const url = 'http://139.162.90.34:11434/api/chat';
const data = {
  model: "llama3.2:3b", // "gemma:2b",
  messages: [
    { role: "user", content: "你是誰？" }
  ]
};

const response = await fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
});

const text = await response.text();
const lines = text.trim().split('\n');
const json1 = '[' + lines.join(',\n') + ']';
const jsonResponse = JSON.parse(json1);

console.log(JSON.stringify(jsonResponse, null, 2));

let r = []
for (let t of jsonResponse) {
    r.push(t.message.content)
}
let responseText = r.join(' ')

console.log(responseText)