import { assertEquals, equal } from "https://deno.land/std@0.63.0/testing/asserts.ts";
export * from "https://deno.land/std@0.63.0/testing/asserts.ts";

export var eq = assertEquals
export var isEq = equal

export function isNear(a:number,b:number,delta:number=0.001) {
  return (Math.abs(a-b) < delta)
}

export function getType(o:any) {
  let t = typeof o
  if (t !== 'object') return t
  return o.constructor.name
}

export function isType(o:any, type:any) {
  if (typeof type === 'string' && getType(o).toLowerCase() === type.toLowerCase()) return true
  if (typeof o === 'object' && o instanceof type) return true
  return false
}

export function isMember(o:any, member:any) {
  if (typeof o === 'string') return member
  if (Array.isArray(o)) return o[o.indexOf(member)]
  if (o instanceof Set && o.has(member)) return member
  if (o instanceof Map) return o.get(member)
  return o[member]
}

export function isContain(o:any, member:any) {
  return isMember(o, member) != null
}
