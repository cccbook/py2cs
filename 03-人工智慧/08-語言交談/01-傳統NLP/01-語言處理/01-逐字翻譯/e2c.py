import sys
e2c = { 'dog':'狗', 'cat':'貓', 'a': '一隻', 'the': '這隻', 'that':'那隻', 'chase':'追', 'eat':'吃' }

def mt(elist): 
    clist = []
    for e in elist:
        c = e2c[e]
        clist.append(c)
    return clist

c = mt(sys.argv[1:])
print(c)
