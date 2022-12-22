from soonerdb import SoonerDB

db = SoonerDB('./db/test.db')

N = 1000
for i in range(N):
    db[f'k{i*2}'] = f'data_{i}'

for i in range(N*2):
    if f'k{i}' in db:
        if i % 100 == 0:
            print(f'k{i}=', db.get(f'k{i}'))
