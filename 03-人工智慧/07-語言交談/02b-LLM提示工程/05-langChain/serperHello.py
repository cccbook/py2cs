import http.client
import json

conn = http.client.HTTPSConnection("google.serper.dev")
payload = json.dumps({
  # "q": "apple inc"
  "q": "langChain 怎麼搭配 groq"
})
headers = {
  'X-API-KEY': 'bad5e8f36ebb8ce93c755ab8b11283ad8152a017',
  'Content-Type': 'application/json'
}
conn.request("POST", "/search", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))
