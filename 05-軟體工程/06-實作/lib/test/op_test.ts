import {eq} from '../test.ts'
import * as O from '../op.ts'

let a = [2,4,6]
let b = [2,2,2]

Deno.test("op:map1", () => {
  eq(O.map1(a,(x)=>x*x), [4,16,36])
})

Deno.test("op:map2", () => {
  eq(O.map2(a,b,(x,y)=>x+y), [4,6,8])
})


