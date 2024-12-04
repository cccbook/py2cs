def IF(cond,job_true,job_false):
    if cond:
        return job_true
    else:
        return job_false

print(f'IF(True,"Yes","No")={IF(True,"Yes","No")}')

# 階層 FACTORIAL(n) = n!
def FACTORIAL(n): 
  return IF(n==0, 1, n*FACTORIAL(n-1))

print(f'FACTORIAL(3)={FACTORIAL(3)}')
