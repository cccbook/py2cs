IF = lambda cond:lambda job_true:lambda job_false:\
    job_true() if cond else job_false()

# 階層 FACTORIAL(n) = n!
def FACTORIAL(n): 
  return IF(n==0)(lambda:1)(lambda:n*FACTORIAL(n-1))

print(f'FACTORIAL(3)={FACTORIAL(3)}')
print(f'FACTORIAL(5)={FACTORIAL(5)}')
