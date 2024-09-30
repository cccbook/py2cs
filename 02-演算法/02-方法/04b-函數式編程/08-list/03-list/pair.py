def pair(x, y):
    return lambda m : m(x, y) # 把 (x,y) 放入閉包中

def head(z):
    return z(lambda p,q : p) # 取出頭部

def tail(z):
    return z(lambda p,q : q) # 取出尾部
