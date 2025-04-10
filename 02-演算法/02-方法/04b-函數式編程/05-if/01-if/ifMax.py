If = lambda cond, a, b: a if cond else b
Max = lambda a,b: If(a>b, a, b)
Min = lambda a,b: If(a<b, a, b)

print('Max(3,5)=', Max(3,5))
print('Min(3,5)=', Min(3,5))
