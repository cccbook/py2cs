# https://medium.com/@adambene/fixed-point-combinators-in-javascript-c214c15ff2f6

U = lambda g: g(g) # U(g) = g(g)

fact = U( # fact = U(g) = g(g) = fact
    lambda g: # g for self-referencing
        lambda x: # currying is for passing the halting condition
            1 if x == 0 else x * fact(x - 1) # recursion
)

print('fact(5)=', fact(5))