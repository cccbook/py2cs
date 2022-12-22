import {expect} from '../expect.ts'

Deno.test("expect", () => {
  expect(3).equal(3)
  expect(3).not.equal(4)
  expect('hello world!').contain('world')
  expect(3).is.a('number')
  expect([1,2,3]).is.a(Array)
})
