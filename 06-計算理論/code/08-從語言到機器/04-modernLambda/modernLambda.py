"""
> (λx.λy.(x y) y)
(λx.λy.(x y) y)
λa.(y a)

?t:?f:t
?t:?f:f
?c:?t:?f:c(t)(f)
(?c:?t:?f:c(t)(f))?t:?f:t

(λc:λt:λf:c(t)(f))λt:λf:t
"""

def run(exp):
    p = parse(exp)
    r = simplify(p)
    return r

def parse(exp):
