# BCD-8421 to 7 segment
import logicmin
t = logicmin.TT(4,7);
# b3 b2 b1 b0  | a b c d e f g
t.add("0000","1111110")
t.add("0001","0110000")
t.add("0010","1101101")
t.add("0011","1111001")
t.add("0100","0110011")
t.add("0101","1011011")
t.add("0110","0011111")
t.add("0111","1110000")
t.add("1000","1111111")
t.add("1001","1110011")
t.add("1010","-------")
t.add("1011","-------")
t.add("1100","-------")
t.add("1101","-------")
t.add("1110","-------")
t.add("1111","-------")
# Outputs minimized independently
sols = t.solve()
print(sols.printN( xnames=['b3','b2','b1','b0'], ynames=['a','b','c','d','e','f','g']))
