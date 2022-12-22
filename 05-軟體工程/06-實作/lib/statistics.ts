const EPSILON = 0.00000001

V.hist = function (a, from = Math.floor(V.min(a)), to=Math.ceil(V.max(a)), step = 1) {
  // from = (from==null) ?  : from
  // to   = (to==null) ?  : to
  var n = Math.ceil((to - from + EPSILON) / step)
  var xc = V.range(from + step / 2.0, to, step)
  var bins = V.array(n, 0)
  let len = a.length
  for (let i=0; i<len; i++) {
    var slot = Math.floor((a[i] - from) / step)
    if (slot >= 0 && slot < n) {
      bins[slot]++
    }
  }
  return {type: 'histogram', xc: xc, bins: bins, from: from, to: to, step: step}
}
