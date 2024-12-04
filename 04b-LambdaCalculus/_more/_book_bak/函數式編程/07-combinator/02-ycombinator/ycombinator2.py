# https://medium.com/@adambene/fixed-point-combinators-in-javascript-c214c15ff2f6

Y = lambda g:g(lambda:Y(g)) # Y(g) = g() = Y(g)

fact = Y( # fact = Y(g) = g() = fact
    lambda g: # g for self-referencing
        lambda x: # this curryed function is returned by g()
            1 if x == 0 else x * fact(x - 1) # g() = fact
)

print('fact(5)=', fact(5))