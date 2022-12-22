from synthesis import *
from functools import reduce
from collections import Counter

def pair(ss, ds):
    """
    Add to `ds` the pair of elements that appears most
    frequently across all sets in `ss`.
    """
    # Collect all pairs of elements found in every set in `ss`.
    ps = [p for s in ss for p in [(x, y) for x in s for y in s if x < y]]

    if len(ps) == 0:
        return (ss, ds, False)
    else:
        # Find the most common pair.
        (p, i) = (Counter(ps).most_common(1)[0][0], len(ds))
        ds.append(p)
        
        # Replace these pairs of elements with an index into
        # the corresponding pair in `ds`.
        ss = [((s - set(p)) | {i}) if set(p).issubset(s) else s for s in ss]

        return (ss, ds, True)

def optimized(xs, vs_ins, vs_outs):
    """
    Synthesize a circuit for the given permutation.
    """
    cache = {}
    def clause(xs, kcs):
        if kcs in cache:
            return cache[kcs]
        elif len(kcs) == 1:
            (k, c) = kcs[0]
            cache[(k, c)] = xs[k] if c == 1 else ~xs[k]
            return cache[(k, c)]
        else:
            (kcs0, kcs1) = parts(kcs, 2)
            cache[kcs] = clause(xs, kcs0) & clause(xs, kcs1)
            return cache[kcs]

    # Construct an initial collection of sets 
    ss = [
        set(r for (r, vo) in enumerate(vs_outs) if vo[c] == 1)
        for c in range(8)
    ]

    # Keep merging the most frequent pair across all sets
    # until there are no pairs left.
    (ds, updated) = (list(range(len(vs_ins))), True)
    while updated:
        (ss, ds, updated) = pair(ss, ds)

    # Take the disjunction of every formula that corresponds
    # to an input vector that maps to `1`.
    cs = [clause(xs, tuple(enumerate(vi))) for vi in vs_ins]
    for (k, (i, j)) in enumerate(ds[len(vs_ins):]):
        cs.append(cs[i] | cs[j])

    return outputs([cs[k] for [k] in ss])

test(optimized)
