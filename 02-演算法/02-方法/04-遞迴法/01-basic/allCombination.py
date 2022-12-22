def combination(A, m): # 從 A 陣列中取出 m 個的所有可能性
	chooses = []
	c(A, len(A), m, chooses, m)

def c(A, n, k, chooses, m): # 從 A[0..n] 中選取 k 個補進 chooses，如果滿 m 個就印出
	if len(chooses)==m:
		print(chooses)
		return
	if n <= 0: return
	c(A,n-1,k,chooses,m) # C(n-1,k) // A[n-1] 沒取到

	chooses.append(A[n-1])
	c(A,n-1,k-1,chooses,m) # C(n-1,k-1) // A[n-1] 有取到
	del chooses[-1]

combination([1,2,3,4,5], 3)

