'''
Jacobi 必需是主對角線絕對優勢的矩陣，否則可能會發散(不收斂)
5x+y  = 6
 x+4y = 5

x = (6-y)/5
y = (5-x)/4

註: 對於 x+2y = 3
        2x+y  = 3

可能會不收斂！
'''

dmin = 0.000000001

def iterate(x, y):
    while (True):
        print('x=', x, 'y=', y)
        xNew = (6-y)/5 # xNew = 3-2*y
        yNew = (5-x)/4 # yNew = 3-2*x
        if abs(xNew-x) < dmin and abs(yNew-y) < dmin: break
        x = xNew
        y = yNew
    return x, y

print('solve: (x,y) = ', iterate(0, 0))
