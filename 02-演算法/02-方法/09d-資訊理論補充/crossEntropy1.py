import math

def log2(x):
    return math.log(x,2)

def entropy(p):
    r = 0
    for i in range(len(p)):
        r += p[i]*log2(1/p[i])
    return r

def cross_entropy(p,q):
    r = 0
    for i in range(len(p)):
        r += p[i]*log2(1/q[i])
    return r

def kl_divergence(p,q):
    r = 0
    for i in range(len(p)):
        r += p[i]*log2(p[i]/q[i])
    return r

p = [1/4,1/4,1/4,1/4]
q = [1/8,1/4,1/4,3/8]
r = [1/100,1/100,1/100,97/100]

print('p=', p)
print('q=', q)
print('r=', r)

print('entropy(p)=', entropy(p))
print('entropy(q)=', entropy(q))
print('entropy(r)=', entropy(r))
print('cross_entropy(p,p)=', cross_entropy(p,p))
print('cross_entropy(p,q)=', cross_entropy(p,q))
print('cross_entropy(p,r)=', cross_entropy(p,r))

print('cross_entropy(p,q)=', cross_entropy(p,q))
print('kl_divergence(p,q)=', kl_divergence(p,q))
print('entropy(p)=', entropy(p))
print('entropy(p)+kl(p,q)=', entropy(p)+kl_divergence(p,q))

