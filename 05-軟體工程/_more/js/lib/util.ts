export function near(a:number,b:number,delta:number=0.001) {
  return (Math.abs(a-b) < delta)
}

export var clone = function (o:any) {
  if (null == o || "object" != typeof o) return o
  if (o.constructor != Object && o.clone != null) return o.clone()
  let r = JSON.parse(JSON.stringify(o)) // 這只會複製非函數欄位！
  if (o.constructor == Object) { // 複製非類別的函數
    for (var attr in o) {
      if (typeof o[attr] === 'function' && o.hasOwnProperty(attr)) r[attr] = o[attr]
    }
  }
  return r
}

export var random = function (min:number=0, max:number=1) {
  return min + Math.random()*(max-min)
}

export var randomInt = function (min:number, max:number) {
  return Math.floor(random(min, max))
}

export var randomChoose = function (a:any[]) {
  return a[randomInt(0, a.length)]
}

export var samples = function (a:any[], n:number) {
  let s = new Array(n)
  for (let i=0; i<n; i++) {
    s[i] = randomChoose(a)
  }
  return s
}

/*
export function defaults(args, defs) {
  let r = Object.assign({}, args)
  for (let k in defs) {
    r[k] = (args[k] == null) ? defs[k] : args[k]
  }
  return r
}
*/
