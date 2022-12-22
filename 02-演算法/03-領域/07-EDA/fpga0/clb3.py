and3   = [0,0,0,0,0,0,0,1]
or3    = [0,1,1,1,1,1,1,1]
xor3   = [0,1,1,0,1,0,0,1]
carry3 = [0,0,0,1,0,1,1,1]
mux2   = [0,0,1,1,0,1,0,1] # mux(s, a, b)

def idx3(a,b,c):
    return a*4+b*2+c

class CLB3:
    def __init__(self, lut=None, reg=None): # clk, out, cin, cout, row, add
        self.lut = lut
        self.reg = reg
    '''
          -Reg-- oReg
    LUT -------- oLut
    '''
    def run(self,a,b,c,rw=0): # li:lut index, rw:reg write
        i = idx3(a,b,c)
        oLut = self.lut[i]
        oReg = self.reg
        if rw: self.reg = oLut
        return oLut, oReg

if __name__ == '__main__':
    sum = CLB3(xor3)
    carry = CLB3(carry3)
    # a, b, c, rw, ro = 0, 1, 1, 0, 0
    a, b, c, rw = 1, 1, 1, 1
    s,_ = sum.run(a,b,c,rw)
    c,_ = carry.run(a,b,c,0)
    print("s=", s, "c=", c, "sum.reg=", sum.reg, "carry.reg=", carry.reg)
