C = [[None]*100 for _ in range(100)]

def c(n, k):
    if k < 0 or k > n: return 0
    if not C[n][k] is None: return C[n][k]
    if k==0 or n <= 1:
        C[n][k] = 1
    else:
        C[n][k] = c(n-1,k) + c(n-1, k-1)
    return C[n][k]

print("c(5,2)=", c(5,2))
print("c(7,3)=", c(7,3))
print("c(12,5)=", c(12,5))
print("c(60,30)=", c(60,30))
