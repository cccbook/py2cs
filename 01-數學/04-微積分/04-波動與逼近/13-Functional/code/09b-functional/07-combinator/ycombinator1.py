# https://medium.com/@adambene/fixed-point-combinators-in-javascript-c214c15ff2f6

Y = lambda g:g(lambda:Y(g)) # Y(g) = g() = Y(g)

fact = Y( # fact = Y(g) = g() = fact
    lambda g: # g for self-referencing
        lambda x: # this curryed function is returned by g()
            1 if x == 0 else x * g()(x - 1) # g() = fact
)

print('fact(5)=', fact(5))

# 令 f = lambda g: lambda x: 1 if x == 0 else x * g()(x - 1)

# fact(5) = f(lambda:Y(f))(5) 
# = 1 if x == 0 else x * g()(x - 1)
#                  = 5 * f()(4) = 5*Y(f)(4)
