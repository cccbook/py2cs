from synthesis import *
from functools import reduce

def naive(xs, vs_ins, vs_outs):
    """
    Synthesize a circuit for the given permutation.
    """
    def clause(xs, kcs):
        if len(kcs) == 1:
            (k, c) = kcs[0]
            return xs[k] if c == 1 else ~xs[k]
        else:
            (kcs0, kcs1) = parts(kcs, 2)
            return clause(xs, kcs0) & clause(xs, kcs1)

    # The set of all clauses, one for each input vector.
    cs = [clause(xs, tuple(enumerate(vi))) for vi in vs_ins]

    # Index sets of input vectors that should be included
    # for each output bit.
    ps = [
        reduce(
            (lambda p, q: p | q),
            [
                clause(xs, tuple(enumerate(vs_ins[r])))
                for (r, vo) in enumerate(vs_outs) if vo[c] == 1
            ]
        )
        for c in range(8)
    ]

    return outputs(ps)

test(naive)