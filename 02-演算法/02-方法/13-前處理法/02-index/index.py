data = ["c", "a", "b", "e", "g", "d"]
index = list(range(len(data)))

index.sort(key=lambda i:data[i])
print('idex=', index)
print('data[index]=', [data[i] for i in index])
