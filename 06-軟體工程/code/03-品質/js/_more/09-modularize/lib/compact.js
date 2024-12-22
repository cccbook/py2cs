// _.compact([0, 1, false, 2, '', 3]), [ 1, 2, 3]
module.exports = function compact(list) {
  const clist = []
  for (let o of list) {
    if (o) clist.push(o)
  }
  return clist
}