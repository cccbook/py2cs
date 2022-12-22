import {eq} from '../test.ts'
import {Complex} from '../complex.ts'

let a = new Complex(1,2)
let b = new Complex(2,1)

Deno.test("complex:op", () => {
  eq(a.add(b), new Complex(3,3))
  eq(a.sub(b), new Complex(-1,1))
  eq(a.mul(b), new Complex(0,5))
})
