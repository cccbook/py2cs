assign = (x,y,z) = (0,1,1)
exp = '(x or y) and (x or not z)'
result = eval(exp)
print('exp=', exp, 'assign=', assign, 'result=', result)
assign = (x,y,z) = (0,1,0)
result = eval(exp)
print('exp=', exp, 'assign=', assign, 'result=', result)
