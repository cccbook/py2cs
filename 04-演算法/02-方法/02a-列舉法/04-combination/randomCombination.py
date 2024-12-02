import random

def randomCombination(pA, k):
	A = pA.copy()
	chooses = []
	for _ in range(k):
		i = random.randrange(0, len(A))
		chooses.append(A[i])
		del A[i]
	chooses.sort()
	return chooses

A = [1,2,3,4,5]
for _ in range(10):
	print(randomCombination(A, 3))
