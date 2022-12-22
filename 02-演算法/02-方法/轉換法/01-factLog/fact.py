def factorial(n):
    p = 1
    for i in range(n):
        p = p * (i+1)
    return p
  
print(factorial(10))
