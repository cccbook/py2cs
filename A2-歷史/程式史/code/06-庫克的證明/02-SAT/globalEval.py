g = globals()
assign = (0,1,1)
(g['x'], g['y'], g['z']) = assign
exp = '(x or not z)'
result = eval(exp)
print('exp=', exp, 'assign=', assign, 'result=', result)
print("g=", g)