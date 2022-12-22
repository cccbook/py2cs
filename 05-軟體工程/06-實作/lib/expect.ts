import {isEq, isNear, isType, isMember} from "./test.ts";

export class Expect {
  o: any
  isNot: boolean
  
  constructor(o:any) {
    this.o = o
    this.isNot = false
  }

  check(cond:boolean) {
    return (cond && !this.isNot) || (!cond && this.isNot)
  }

  pass(f:(x:any)=>boolean) {
    let cond = f(this.o)
    if (this.check(cond)) return this
    throw Error('Expect.pass fail!')
  }

  each(f:(x:any)=>boolean) {
    let o = this.o
    for (let k in o) {
      if (!f(o[k])) throw Error('Expect.each fail! key='+k)
    }
  }

  any(f:any) {
    let o = this.o
    for (let k in o) {
      if (f(o[k])) return true
    }
    throw Error('Expect.any fail!')
  }

  equal(o:any) {
    if (this.check(isEq(this.o, o))) return this
    throw Error('Expect.equal fail!')
  }

  near(n:number, gap:number=0.001) {
    if (this.check(isNear(this.o, n, gap))) return this
    throw Error('Expect.near fail!')
  }

  type(type:any) { 
    if (this.check(isType(this.o, type))) return this
    throw Error('Expect.type fail!')
  }

  a(type:any) {
    if (this.check(isType(this.o, type) || isEq(this.o, type))) return this
    throw Error('Expect.a fail!')
  }

  contain(member:any) {
    let m = isMember(this.o, member)
    if (this.check(m != null)) { this.o = m; return this }
    throw Error('Expect.contain fail!')
  }

  get not() {
    this.isNot = !this.isNot
    return this
  }
  get to() { return this }
  get but() { return this }
  get at() { return this }
  get of() { return this }
  get same() { return this }
  get does() { return this }
  get do() { return this }
  get still() { return this }
  get is() { return this }
  get be() { return this }
  get should() { return this }
  get has() { return this }
  get have() { return this }
  get been() { return this }
  get that() { return this }

  get and() { return this }
}

export function expect(o:any) {
  return new Expect(o)
}

/*
  get 那個() { return this.that }
  get 不() { return this.not }
let p = Expect.prototype

p.property = p.contain
p.include = p.contain
p.all = p.each
*/
/*
 = E.希望 = E.願 = E.驗證 = E.確認 = 

p.包含 = p.include
p.通過 = p.pass
p.每個 = p.each
p.等於 = p.equal
p.靠近 = p.near
p.型態 = p.type
p.是 = p.a
p.有 = p.contain
p.屬性 = p.property

*/
