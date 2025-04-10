# Full-adder
import logicmin
# truth table 3 inputs, 2 outputs
t = logicmin.TT(3,2);
# add rows to the truth table (input, ouutput)
# ci a b  |  s co
t.add("000","00")
t.add("001","10")
t.add("010","10")
t.add("011","01")
t.add("100","10")
t.add("101","01")
t.add("110","01")
t.add("111","11")
# minimize functions and get
# solution for analysis and print
sols = t.solve()
# print solution mapped to var names (xnames=inputs, ynames=outputs)
# add debug information
print(sols.printN(xnames=['Ci','a','b'],ynames=['s','Co'], info=True))