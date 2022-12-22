from parts import parts
from circuit import *
from circuitry import *

from itertools import product
from random import shuffle
from tqdm import tqdm

def test(synthesis):
    # Create a permutation of all 8-bit vectors.
    vs_original = list(product(*[[0,1]]*8))
    vs_permuted = list(product(*[[0,1]]*8))
    shuffle(vs_permuted, lambda: 0.5)

    # Execute the synthesis function that is being tested.
    # A synthesis function must accept as it inputs an
    # initial vector to evaluate while constructing the
    # circuit (as necessatied by the `circuitry` library),
    # the original list of vectors, and a permuted list of
    # vectors.
    bit.circuit(circuit())
    bs = synthesis(
        bits([input(i) for i in ([0]*8)]),
        vs_original,
        vs_permuted
    )

    # Display some statistics and whether the circuit
    # correctly implements the permutation.
    c = bit.circuit()
    checks = ([
        (vo == tuple(c.evaluate(list(vi))))
        for (vi, vo) in tqdm(
            list(zip(vs_original, vs_permuted)),
            position=0, leave=True
        )
    ])
    print(all(checks))
    print({o: c.count(lambda g: g.operation == o) for o in [op.and_, op.or_, op.not_]})

    