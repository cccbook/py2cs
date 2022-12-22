async function myfetch(url, options) {
  return await fetch(url, options)
}

// Deno.args[0]
let fd = new FormData()
fd.append("name", "ccc")
fd.append("age", "53")
const textResponse = await myfetch('http://127.0.0.1:8080/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
  },
  body: fd,
})
const textData = await textResponse.text();
console.log(textData);

