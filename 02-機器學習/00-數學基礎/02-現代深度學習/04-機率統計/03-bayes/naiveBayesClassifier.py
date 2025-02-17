from naiveBayesProb import prob, naiveBayesProb

def argmax(list):
    max = -999999
    index = None
    for k in range(len(list)):
        if list[k] > max:
            index=k
            max=list[k]
    return index

f = ['f1', 'f2']
c = ['c1', 'c2']
p = list(map(lambda ci : naiveBayesProb(prob, ci, f), c))
print('p=', p)

for i in range(len(c)):
    print('P({}|f1,f2) = {}'.format(c[i], p[i]))

imax = argmax(p)
print('{} 的機率最大'.format(c[imax]))
