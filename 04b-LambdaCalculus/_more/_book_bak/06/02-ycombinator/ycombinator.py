Y_COMBINATOR = lambda f: (lambda x: f(lambda v: x(x)(v)))(lambda x: f(lambda v: x(x)(v)))

FACTORIAL = Y_COMBINATOR(lambda g: lambda n: 
    1 if (n==0) else n*g(n-1)
)

print(f'FACTORIAL(5)={FACTORIAL(5)}')
