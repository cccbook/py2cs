data = ["c", "a", "b", "e", "g", "d"]
N = 17
hash_table = [None]*N

for d in data:
    i = hash(d)%N
    while hash_table[i] is not None:
        i = (i+1)%N
    hash_table[i] = d

print('hash_table=', hash_table)
