# P(x,y,z) = P(x) * P(y) * P(z)
def naiveProb(prob, list):
  p = 1
  for e in list:
      p = p * prob[e]
  return p

prob = {
  'x': 0.5,
  'y': 0.2,
  'z': 0.3
}

print('P(x,y,z) = P(x)P(y)P(z)=', naiveProb(prob, ['x','y','z']))
# 前提是 x,y,z 三個隨機變數都條件獨立。 （所以說是天真）
