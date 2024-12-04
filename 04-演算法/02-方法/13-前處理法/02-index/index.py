data = ["c", "a", "b", "e", "g", "d"]
index = list(range(len(data)))
print('index=', index)

index.sort(key=lambda i:data[i])
print('index=', index)
print('data[index]=', [data[i] for i in index])
