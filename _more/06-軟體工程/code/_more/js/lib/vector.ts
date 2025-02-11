import * as U from './util.ts'

export function vector(n:number, value:number=0) {
  let a = new Array(n)
  return a.fill(value)
}

export function near(a:number[], b:number[], delta=0.001) {
  if (a.length != b.length) return false
  let len = a.length
  for (var i = 0; i < len; i++) {
    if (Math.abs(a[i]-b[i]) > delta) return false
  }
  return true
}

export function range(begin:number, end:number, step:number=1) {
  let len = Math.floor((end-begin)/step)
  let a = new Array(len)
  let i = 0
  for (let t=begin; t<end; t+=step) {
    a[i++] = t
  }
  return a
}

export let steps = range

export function op2(op:string) { // 這個函數強調速度，所以會比較長 ...
  let text = `
  var ai, bi, i, c
  let aA = Array.isArray(a)
  let bA = Array.isArray(b)
  let len = a.length || b.length
  if (aA && bA) {
    c = new Array(len)
    for (let i=0; i<len; i++) {
      ai=a[i]
      bi=b[i]
      c[i] = ${op}
    }
    return c
  }
  if (!aA && !bA) { ai=a; bi=b; return ${op} }
  c = new Array(len)
  for (let i=0; i<len; i++) {
    ai=(aA) ? a[i] : a
    bi=(bA) ? b[i] : b
    c[i] = ${op}
  }
  return c
  `
  return new Function('a', 'b', text)
}

export var add = op2('ai+bi')
export var sub = op2('ai-bi')
export var mul = op2('ai*bi')
export var div = op2('ai/bi')
export var mod = op2('ai%bi')
export var pow = op2('Math.pow(ai,bi)')
export var and = op2('ai&&bi')
export var or  = op2('ai||bi')
export var xor = op2('(ai || bi) && !(ai && bi)')
export var band= op2('ai&bi')
export var bor = op2('ai|bi')
export var bxor= op2('ai^bi')
export var eq  = op2('ai==bi')
export var neq = op2('ai!=bi')
export var lt  = op2('ai<bi')
export var gt  = op2('ai>bi')
export var leq = op2('ai<=bi')
export var geq = op2('ai>=bi')

// Uniary Operation
export function op1(op:string) {
  let text = `
  var ai
  let aA = Array.isArray(a)
  if (!aA) { ai=a; return ${op} }
  let len = a.length
  let c = new Array(len)
  for (let i=0; i<len; i++) {
    ai=a[i]
    c[i] = ${op}
  }
  return c
  `
  return new Function('a', text)
}

export var neg = op1('-ai')
export var abs = op1('Math.abs(ai)')
export var log = op1('Math.log(ai)')
export var not = op1('!ai')
export var sin = op1('Math.sin(ai)')
export var cos = op1('Math.cos(ai)')
export var tan = op1('Math.tan(ai)')
export var cot = op1('Math.cot(ai)')
export var sec = op1('Math.sec(ai)')
export var csc = op1('Math.csc(ai)')
export var asin= op1('Math.asin(ai)')
export var acos= op1('Math.acos(ai)')
export var atan= op1('Math.atan(ai)')
export var atan2=op1('Math.atan2(ai)')
export var atanh=op1('Math.atanh(ai)')
export var cbrt= op1('Math.cbrt(ai)')
export var ceil= op1('Math.ceil(ai)')
export var clz32=op1('Math.clz32(ai)')
export var cosh= op1('Math.cosh(ai)')
export var exp = op1('Math.exp(ai)')
export var expm1= op1('Math.expm1(ai)')
export var floor= op1('Math.floor(ai)')
export var fround= op1('Math.fround(ai)')
export var hypot= op1('Math.hypot(ai)')
export var imul= op1('Math.imul(ai)')
export var log10= op1('Math.log10(ai)')
export var log1p= op1('Math.log1p(ai)')
export var log2= op1('Math.log2(ai)')
export var round= op1('Math.round(ai)')
export var sign= op1('Math.sign(ai)')
export var sqrt= op1('Math.sqrt(ai)')
export var trunc= op1('Math.trunc(ai)')

// 累積性運算
export var dot = function (a:number[],b:number[]) {
  let len = a.length
  let r = 0
  for (let i=0; i<len; i++) {
    r += a[i] * b[i]
  }
  return r
}

export var min = function (a:number[]) {
  let len = a.length, r = a[0]
  for (let i=1; i<len; i++) {
    if (a[i] < r) r = a[i]
  }
  return r
}

export var max = function (a:number[]) {
  let len = a.length, r = a[0]
  for (let i=1; i<len; i++) {
    if (a[i] > r) r = a[i]
  }
  return r
}

export var any = function (a:number[]) {
  let len = a.length
  for (let i=0; i<len; i++) {
    if (a[i]) return true
  }
  return false
}

export var all = function (a:number[]) {
  let len = a.length
  for (let i=0; i<len; i++) {
    if (!a[i]) return false
  }
  return true
}

export var sum = function(a:number[]) {
  let len = a.length
  let r = 0
  for (let i=0; i<len; i++) {
    r += a[i]
  }
  return r
}

export var product = function(a:number[]) {
  let len = a.length
  let r = 1
  for (let i=0; i<len; i++) {
    r *= a[i]
  }
  return r
}

export var norm = function (a:number[]) {
  let a2 = pow(a, 2)
  return Math.sqrt(sum(a2))
}

export var norminf = function (a:number[]) {
  let len = a.length
  let r = 0
  for (let i=0; i<len; i++) {
    r = Math.max(r, Math.abs(a[i]))
  }
  return r
}

// norminf: ['accum = max(accum,abs(xi));','var accum = 0, max = Math.max, abs = Math.abs;'],

export var mean = function(a:number[]) {
  return sum(a)/a.length
}

export var sd = function (a:number[]) {
  let m = mean(a)
  let diff = sub(a, m)
  let d2 = pow(diff, 2)
  return Math.sqrt(sum(d2)/(a.length-1))
}

// V.range = uu6.range
// V.steps = uu6.steps

export var random = function (r:number[], min:number=0, max:number=1) {
  let len = r.length
  for (let i=0; i<len; i++) {
    r[i] = U.random(min, max)
  }
}

/*
export var assign = function (r:number[], o) {
  let isC = (typeof o === 'number')
  if (!isC) uu6.be(r.length === o.length)
  let len = r.length
  for (let i=0; i<len; i++) {
    r[i] = isC ? o : o[i]
  }
  return r
}
*/
export var normalize = function (r:number[]) {
  let ar = abs(r)
  let s = sum(ar) // 不能用 sum，sum 只適用於機率。
  let len = r.length
  for (let i=0; i<len; i++) {
    r[i] = (s==0) ? 0 : r[i]/s
  }
  return r
}

export var normalize2 = function (r:number[]) {
  let norm2 = norm(r)
  if (norm2 === 0) return r
  let len = r.length
  for (let i=0; i<len; i++) {
    r[i] = r[i]/norm2
  }
  return r
}

/*


export function dot(a:number[], b:number[]) {
  let sum = 0
  let len = a.length
  for (var i = 0; i < len; i++) {
    sum += a[i] * b[i] // 速度較快
  }
  return sum
}

export function add(a:number[], b:number[]) {
  let len = a.length
  let r = new Array(len)
  for (var i = 0; i < len; i++) {
    r[i] = a[i] + b[i]
  }
  return r
}

export function sub(a:number[], b:number[]) {
  let len = a.length
  let r = new Array(len)
  for (var i = 0; i < len; i++) {
    r[i] = a[i] - b[i]
  }
  return r
}

export function mul(a:number[], b:number[]) {
  let len = a.length
  let r = new Array(len)
  for (var i = 0; i < len; i++) {
    r[i] = a[i] * b[i]
  }
  return r
}

export function div(a:number[], b:number[]) {
  let len = a.length
  let r = new Array(len)
  for (var i = 0; i < len; i++) {
    r[i] = a[i] / b[i]
  }
  return r
}

export function addc(a:number[], c:number) {
  let len = a.length
  let r = new Array(len)
  for (var i = 0; i < len; i++) {
    r[i] = a[i] + c
  }
  return r
}

export function subc(a:number[], c:number) {
  return addc(a, -c)
}

export function mulc(a:number[], c:number) {
  let len = a.length
  let r = new Array(len)
  for (var i = 0; i < len; i++) {
    r[i] = a[i] * c
  }
  return r
}

export function divc(a:number[], c:number) {
  return mulc(a, 1/c)
}
*/