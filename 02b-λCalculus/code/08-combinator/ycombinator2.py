# js 版來源 -- https://bmsdave.github.io/blog/y-combinator-eng/

# Y(f) = f(Y(f)) = f(f(Y(f))) ..., 但是要 lazy ，所以最後加上 x
Y = lambda f: lambda x:f(Y(f))(x)

factorial = lambda f: lambda n: 1 if n == 0 else n * f(n - 1)

print(f'Y(factorial)(5)={Y(factorial)(5)}')
