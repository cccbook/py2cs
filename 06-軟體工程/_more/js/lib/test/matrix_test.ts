import {eq, assert} from '../test.ts'
import * as V from '../vector.ts'
import * as M from '../matrix.ts'
import * as U from '../util.ts'

let a = [[1,2],[3,4]]
var at:number[][]

Deno.test("matrix: transpose", () => {
  at = M.transpose(a)
  eq(at, [[1,3], [2,4]])
})

Deno.test("matrix: diag", () => {
  let D = M.diag([1,2,3])
  eq(M.flatten(D), [1,0,0, 0,2,0, 0,0,3])
})

Deno.test("matrix: identity", () => {
  let I = M.identity(3)
  eq(M.flatten(I), [1,0,0, 0,1,0, 0,0,1])
})

Deno.test("matrix: dot", () => {
  let aat = M.dot(a, at)
  eq(aat, [[5,11], [11,25]])
})

Deno.test("matrix: inv", () => {
  let b = M.inv(a)
  let ab = M.dot(a, b)
  assert(V.near(M.flatten(ab), [1,0, 0,1]))
})

Deno.test("matrix: det", () => {
  let d = M.det(a)
  assert(U.near(d, -2))
})

Deno.test("matrix: LU", () => {
  let lup = M.lu(a)
  // console.log('lup=', lup)
  assert(V.near(M.flatten(lup.LU), [3, 4, 0.3333, 0.6667]))
  let b = [17, 39]
  let x = [5, 6]
  let s = M.luSolve(lup, b)
  // console.log('s=', s)
  assert(V.near(s, x))
})


Deno.test("matrix: SVD", () => {
  let svd = M.svd(a)
  // console.log('svd=', svd)
  let Ut = M.transpose(svd.U)
  let Vt = M.transpose(svd.V)
  let UtU = M.dot(Ut, svd.U)
  let VVt = M.dot(svd.V, Vt)
  // console.log('UtU=', UtU)
  // console.log('VVt=', VVt)
  assert(V.near(M.flatten(UtU), M.flatten(M.identity(svd.U.length))))
  assert(V.near(M.flatten(VVt), M.flatten(M.identity(svd.V.length))))
  let US = M.dot(svd.U, M.diag(svd.S))
  let USVt = M.dot(US, Vt)
  // console.log('USV=', USVt)
  assert(V.near(M.flatten(USVt), M.flatten(a)))
})
