// var array = [1];
// var other = _.concat(array, 2, [3], [[4]]);
// other equalTo([1, 2, [3], [[4]]])
module.exports = function concat() {
  const clist = arguments[0].slice(0)
  for (let i=1; i<arguments.length; i++) {
    clist.push(arguments[i])
  }
  return clist
}
