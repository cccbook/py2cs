def c(n, k):
    if k < 0 or k > n: return 0
    # if k > n-k: k = n - k
    if k==0 or n <= 1: return 1
    return c(n-1, k) + c(n-1, k-1)

print("c(5,2)=", c(5,2))
print("c(7,3)=", c(7,3))
print("c(12,5)=", c(12,5))
print("c(60,30)=", c(60,30))
