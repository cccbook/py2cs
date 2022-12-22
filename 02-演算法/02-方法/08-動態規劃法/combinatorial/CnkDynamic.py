# http://mathworld.wolfram.com/PascalsFormula.html
# https://en.wikipedia.org/wiki/Pascal%27s_rule
# https://en.wikipedia.org/wiki/Pascal%27s_triangle
# https://en.wikipedia.org/wiki/Binomial_coefficient
'''
c(n, k) = 1                        , if k = 0 or k = n
        = c(n-1, k-1) + c(n-1, k)  , if k <= n-k
'''

def c(N, K):
    C = [None]*(N+1) 
    for n in range(N+1):
        C[n] = [0]*(N+1)
        C[n][0] = 1
        C[n][n] = 1
    print("C=", C)
    for n in range(N):
        for k in range(n):
            C[n+1][k+1] = C[n][k] + C[n][k+1]

    for n in range(N+1):
        print("C[", n, "]=", C[n])

    return C[N][K] 


print("c(5,2)=", c(5,2))

'''
print("c(7,3)=", c(7,3))
print("c(12,5)=", c(12,5))
print("c(60,30)=", c(60,30))
'''