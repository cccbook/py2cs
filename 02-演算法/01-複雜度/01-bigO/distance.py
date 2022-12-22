def distance(p1, p2):
    dx = p1['x'] - p2['x']
    dy = p1['y'] - p2['y']
    return dx*dx+dy*dy

p1 = {'x':3, 'y':4}
p2 = {'x': 0, 'y':0}

print(f'distance({p1},{p2})={distance(p1,p2)}')
