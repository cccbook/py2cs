// ================== Map Reduce =========================
export function map1(a:any, f:(x:any)=>any) {
  if (a instanceof Array) {
    var fa = new Array(a.length)
    for (var i = 0; i < a.length; i++) {
      fa[i] = map1(a[i], f)
    }
    return fa
  } else {
    return f(a)
  }
}

export function map2(a:any, b:any, f:(x:any, y:any)=>any) {
  if (a instanceof Array) {
    var fa = new Array(a.length)
    var isArrayB = (b instanceof Array)
    for (var i = 0; i < a.length; i++) {
      var bi = isArrayB ? b[i] : b
      fa[i] = map2(a[i], bi, f)
    }
    return fa
  } else {
    return f(a, b)
  }
}

export function reduce(a:any, f:any, init:(x:any)=>any) {
  var result = init
  if (a instanceof Array) {
    for (var i in a) {
      result = f(result, reduce(a[i], f, init))
    }
  } else {
    result = f(result, a)
  }
  return result
}


/*
const uu6 = require('uu6')
const V = module.exports = {}


class Vector {
  constructor(o) { this.v = (Array.isArray(o))?o.slice(0):new Array(o) }
  static random(n, min=0, max=1) { return new Vector(V.random(n, min, max)) }
  static range(begin, end, step=1) { return new Vector(V.range(begin, end, step)) }
  static zero(n) { return new Vector(n) }
  assign(o) { let a=this; V.assign(a.v, o); return a }
  random(min=0, max=1) { this.v = V.random(n, min, max) }
  add(b) { let a=this; return a.clone(V.add(a.v,b.v)) }
  sub(b) { let a=this; return a.clone(V.sub(a.v,b.v)) } 
  mul(b) { let a=this; return a.clone(V.mul(a.v,b.v)) } 
  div(b) { let a=this; return a.clone(V.div(a.v,b.v)) } 
  mod(b) { let a=this; return a.clone(V.mod(a.v,b.v)) }
  neg() { let a=this; return a.clone(V.neg(a.v)) }
  dot(b) { let a=this; return V.dot(a.v,b.v) }
  min() { let a=this; return V.min(a.v) }
  max() { let a=this; return V.max(a.v) }
  sum() { let a=this; return V.sum(a.v) }
  norm() { let a=this; return V.norm(a.v)  }
  mean() { let a=this; return V.mean(a.v) }
  sd() { let a=this; return V.sd(a.v) }
  toString() { return this.v.toString() }
  clone(v) { return new Vector(v||this.v) }
  hist(from, to, step) { let a = this; return V.hist(a.v, from, to, step) }
  get length() { return this.v.length }
}

V.Vector = Vector

V.vector = function (o) {
  return new Vector(o)
}


// ================= 以下函數針對二維向量 ====================

// 叉積 a0*b1-a1*b0 (只適用於二維向量)
V.cross = function (a,b) {
  return a[0]*b[1]-a[1]*b[0]
}

// 偽極角 : 針對二維陣列
V.pseudoPolarAngle = function (v, v0) {
  uu6.be(v.length === 2 && v0.length === 2)
  let vd = V.sub(v, v0)
  let [x, y] = V.normalize(vd)
  // console.log('x,y=', x, y)
  if (x>=0 && y>=0) return y
  if (x<0 && y >=0) return (Math.PI/2) - x
  if (x<0 && y<0) return Math.PI-y
  return Math.PI*3/2+x
}
*/