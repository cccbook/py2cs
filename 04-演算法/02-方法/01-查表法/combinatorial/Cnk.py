def factorial(n):
    p = 1
    for i in range(n):
        p = p * (i+1)
    return p

def c(n, k):
    return factorial(n) / (factorial(k)*factorial(n-k))

print("c(5,2)=", c(5,2))
print("c(7,3)=", c(7,3))
print("c(12,5)=", c(12,5))
print("c(60,30)=", c(60,30))
