function objToUrl(obj) {
  let r = []
  for (let key in obj) {
    r.push(`${key}=${obj[key]}`)
  }
  return r.join('&')
}

async function myfetch(url, options) {
  return await fetch(url, options)
}

let p = {name:"ccc", age:53}
const textResponse = await myfetch('http://127.0.0.1:8080/', {
  method: 'POST',
  headers: {
    'Content-Type': 'text/plain; charset=UTF-8',
  },
  body: objToUrl(p) // JSON.stringify(p),
})
const textData = await textResponse.text();
console.log(textData);

