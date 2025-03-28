# js -- https://gist.github.com/Avaq/1f0636ec5c8d6aed2e45?permalink_comment_id=3908358

I  = lambda x: x
K  = lambda x: lambda y:  x        # 丟棄 y
A  = lambda f: lambda x: f (x)
T  = lambda x: lambda f: f (x)
W  = lambda f: lambda x: f (x) (x) # 重複
C  = lambda f: lambda y: lambda x: f (x) (y)  # 交換
B  = lambda f: lambda g: lambda x: f (g (x))  # Compose = f*g
S  = lambda f: lambda g: lambda x: f (x) (g(x))
S_ = lambda f: lambda g: lambda x: f (g(x)) (x)
S2 = lambda f: lambda g: lambda h: lambda x: f (g(x)) (h(x))
P  = lambda f: lambda g: lambda x: lambda y: f (g(x)) (g(y))
Y  = lambda f: (lambda g: g(g)) (lambda g: f (lambda x: g (g) (x)))

# f(fixpoint) = fixpoint
f = lambda x:x
# assert Y(f) == f(Y(f))

factorial = lambda f: lambda n: 1 if n == 0 else n * f(n - 1);

# Y(f) = f(Y(f)) = f(f(Y(f))) ...
Y = lambda f: lambda x:f(Y(f))(x)

print(f'Y(factorial)(5)={Y(factorial)(5)}');
