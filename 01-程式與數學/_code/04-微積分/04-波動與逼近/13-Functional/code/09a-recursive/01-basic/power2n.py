def power2(n):
    if n == 0:
        return 1
    # return power2(n-1) + power2(n-1) 
    return 2*power2(n-1)

print('power2(10)=', power2(10))
print('power2(100)=', power2(100))
