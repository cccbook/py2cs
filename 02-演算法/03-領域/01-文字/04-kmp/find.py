def find(s, k):
	klen = len(k)
	for i in range(len(s)):
		if s[i:i+klen]==k:
			return i
	return -1

s = "ababababababcabab"
k = "abababc"
print('find(s, k)=', find(s, k))
