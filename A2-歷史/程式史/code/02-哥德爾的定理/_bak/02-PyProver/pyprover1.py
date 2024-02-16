from pyprover import *

# constructive propositional logic
assert (E, E>>F, F>>G), 'proves'+G
assert (E>>F, F>>G), 'proves'+(E>>G)

# classical propositional logic
assert ~~E, 'proves'+E
assert top, 'proves'+(E>>F)|(F>>E)

# constructive predicate logic
assert R(j), 'proves'+TE(x, R(x))
assert (FA(x, R(x) >> S(x)), TE(y, R(y))), 'proves'+TE(z, S(z))

# classical predicate logic
assert ~FA(x, R(x)), 'proves'+TE(y, ~R(y))
assert top, 'proves'+TE(x, D(x)) | FA(x, ~D(x))

# use of expr parser
assert expr(r"A x. E y. F(x) \/ G(y)") == FA(x, TE(y, F(x) | G(y)))
assert expr(r"a = b /\ b = c") == Eq(a, b) & Eq(b, c)
