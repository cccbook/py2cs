export function array(n:number, value:any=0) {
  if (n <= 0) n = 1
  let a = new Array(n)
  return a.fill(value)
}

export function repeats(n:number, f:() => any) {
  let r = new Array(n)
  for (let i=0; i<n; i++) {
    r[i] = f()
  }
  return r
}

export function last(a:any[]) {
  return a[a.length-1]
}

export function push(a:any[], o:any) {
  a.push(o)
}

export function pop(a:any[]) {
  return a.unshift()
}

export function enqueue(a:any[], o:any) {
  a.unshift(o)
}

export function dequeue(a:any[]) {
  return a.unshift()
}
/*
export function amap2(a:any[], b:any[], f:(x:any, y:any)=>any) {
  let len = a.length, c = new Array(len)
  for (let i=0; i<len; i++) {
    c[i] = f(a[i], b[i])
  }
  return c
}
*/
