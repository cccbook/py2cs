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

let p = {title:"zzz", body:"zzzzzz"}
const textResponse = await myfetch('http://127.0.0.1:8000/post', {
  method: 'POST',
  headers: {
    // 'Content-Type': 'text/plain; charset=UTF-8',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
  },
  body: objToUrl(p) // JSON.stringify(p),
})
const textData = await textResponse.text();
console.log(textData);

