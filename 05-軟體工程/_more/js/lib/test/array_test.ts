import {eq} from '../test.ts'
import * as A from '../array.ts'

Deno.test("array", () => {
  eq(A.array(3, "x"), ["x","x","x"])

})
