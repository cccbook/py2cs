citys = [
    (0,3),(0,0),
    (0,2),(0,1),
    (1,0),(1,3),
    (2,0),(2,3),
    (3,0),(3,3),
    (3,1),(3,2)
]

#path = [i for i in range(len(citys))]
l = len(citys)
path = [(i+1)%l for i in range(l)]
print(path)

def distance(p1, p2):
    print('p1=', p1)
    x1, y1 = p1
    x2, y2 = p2
    return ((x2-x1)**2+(y2-y1)**2)**0.5

def pathLength(p):
    dist = 0
    plen = len(p)
    for i in range(plen):
        # dist += distance(citys[p[i]], citys[p[(i+1)%plen]])
        dist += distance(citys[i], citys[p[i]])
    return dist

print('pathLength=', pathLength(path))

