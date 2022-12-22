const argmax = require('../lib/argmax')
# P(C|F1...Fn) = P(C) * P(F1|C) * ....* P(Fn|C)
def naiveBayesProb(prob, c, f) :
   p = prob[c]
  for ( fi of f) p = p*prob[c+'=>'+fi]
  return p


const prob = :
  'c1': 0.6, 'c2': 0.4,
  'c1=>f1': 0.5, 'c1=>f2': 0.8, 'c1=>f3': 0.6,
  'c2=>f1': 0.7, 'c2=>f2': 0.6, 'c2=>f3': 0.2,


const f = ['f1', 'f2']
const c = ['c1', 'c2']
const p = c.map((ci) => naiveBayesProb(prob, ci, f))
for ( i=0  i<c.length  i++) :
  print('P(%s|f1,f2) = ', c[i], p[i].toFixed(3))

const :max:pmax, index: imax = argmax(p)
print('%s 的機率最大', c[imax])
