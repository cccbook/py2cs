import json

def editDistance (b, a):
    alen,blen = len(a), len(b)
    if alen==0: return blen
    if blen==0: return alen

    m = [0]*(blen+1)
    for i in range(blen+1):
        m[i] = [0]*(alen+1)
        m[i][0] = i
    for j in range(alen+1):
        m[0][j] = j
    for i in range(1,blen+1):
        for j in range(1,alen+1):
            if b[i-1] == a[j-1]:
                m[i][j] = m[i-1][j-1]
            else:
                m[i][j] = min(
                  m[i-1][j-1] + 1, # 取代
                  min(
                    m[i][j-1] + 1, # 插入
                    m[i-1][j] + 1
                  )
                ) # 刪除
    return {'d': m[blen][alen], 'm': m} 

def align (b, a, m) :
    i,j = len(b), len(a)
    ax, bx = '', ''
    while (i > 0 and j > 0):
        if m[i][j] == m[i-1][j]+1:
            # 插入 b[i]
            i-=1
            ax = ' ' + ax
            bx = b[i] + bx
        elif m[i][j] == m[i][j-1] + 1:
            # 插入 a[j]
            j-=1
            ax = a[j] + ax
            bx = ' ' + bx
        elif m[i][j] == m[i-1][j-1] + 1 or m[i][j] == m[i-1][j-1]: # 取代
            # 相同
            i-=1
            j-=1
            bx = b[i] + bx
            ax = a[j] + ax

    while (i > 0) :
        i-=1
        bx = b[i] + bx
        ax = ' ' + ax
    
    while (j > 0) :
        j-=1
        ax = a[j] + ax
        bx = ' ' + bx
    
    print('bx=', bx)
    print('ax=', ax)


def dump (m) :
  for row in m:
    print(json.dumps(row))

'''
a = "010100001"
b = "010100010"
print("dist(%s,%s) = %s", a, b, editDistance(a,b))
'''
# b = "ATG  ATCCG"
a = 'ATGCAATCCC'
b = 'ATGATCCG'
# b = "  TCCGAA"
# a   = "ATCCCAAA"
# b   = "TCCGAA"
e = editDistance(b, a)
print(f'editDistance({b},{a}) = {e["d"]}')
print('====m======\n')
dump(e['m'])
print('===========\n')
align(b, a, e['m'])
