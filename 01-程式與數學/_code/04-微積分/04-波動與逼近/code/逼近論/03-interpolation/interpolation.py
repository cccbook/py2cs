def linear(x0, x1, y0, y1, x):
    y = y0+(x-x0)*(y1-y0)/(x1-x0)
    return y

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
def B(x, k, i, t):
   if k == 0:
      return 1.0 if t[i] <= x < t[i+1] else 0.0
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
   return c1 + c2

def bspline(x, t, c, k):
   n = len(t) - k - 1
   assert (n >= k+1) and (len(c) >= n)
   return sum(c[i] * B(x, k, i, t) for i in range(n))

x0, x1 = 0, 2
y0, y1 = 5, 10
x = 1
print(f'linear({x})={linear(x0, x1, y0, y1, x)}')

k = 2
t = [0, 1, 2, 3, 4, 5, 6]
c = [-1, 2, 0, -1]
print('bspline(2.5, t, c, k)=', bspline(2.5, t, c, k))
