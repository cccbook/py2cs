import random

def permutation(pA):
    A = pA.copy()
    for _ in range(random.randrange(0, len(A)*10)):
        i = random.randrange(0, len(A))
        j = random.randrange(0, len(A))
        A[i], A[j] = A[j], A[i]
    return A

A = ["a", "b", "c"]
for _ in range(100):
    print(permutation(A))

