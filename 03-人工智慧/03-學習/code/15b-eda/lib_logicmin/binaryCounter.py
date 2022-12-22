# Finite-state machine
# x=0 => hold
# x=1 => binary up count
# y = 1 in states: e1 and e3
import logicmin
# state labels
states = ['e0','e1','e2','e3']
# 2 bits for state codes
# 1 input variable
# 1 output variable
m = logicmin.FSM(states,2,1,1)
# transition table
m.add('0','e0','e0','0')
m.add('1','e0','e1','0')
m.add('0','e1','e1','1')
m.add('1','e1','e2','1')
m.add('0','e2','e2','0')
m.add('1','e2','e3','0')
m.add('0','e3','e3','1')
m.add('1','e3','e0','1')
# asign code to states
codes = {'e0':0,'e1':1,'e2':2,'e3':3}
m.assignCodes(codes)
# solve with D flip-flops
sols = m.solveD()
# print solution with input and output names
print(sols.printN(xnames=['X','Q1','Q0'], ynames=['D1','D0','Y']))
