# 函數 f 對變數 k 的偏微分: partial(p) / dk
 partial = def (f, p, k, h=0.01) :
   p1 = Object.assign(:, p)
  p1[k] += h
  return (f(p1) - f(p)) / h


 f = def (p) :
   x = p.x, y = p.y
  return x*x + y*y


print('df(x:3,y:2)/dx=', partial(f, :x:3, y:2, 'x'))
print('df(x:3,y:2)/dy=', partial(f, :x:3, y:2, 'y'))

