# 參考 -- https://courses.cs.washington.edu/courses/cse467/03wi/FPGA.pdf
# FPGA = CLBs + routes
# CLB = LUT*2 + FullAdder + Flip-Flop
# 於是只要指定 route + 每個 CLB 的 LUT 內容，就能完成 FPGA 的燒錄。
# Clock 呢? 就用 一次 clock() 函數呼叫當成一個 Clock 吧！
class FPGA:
    def __init__(self, nrow, ncol, nLutIn):
        self.blocks = []
        for _ in range(nrow):
            row = []
            for _ in range(ncol):
                row.append(LogicBlock(nLutIn))
            self.blocks.append(row)
        self.route = [[[0]*nLutIn for _ in range(ncol)] for _ in range(nrow)]

'''
class LUT:
    def __init__(self, outs):
        self.outs = outs
    def lookup(i):
        return self.outs[i]

class RAM: # 在此 FPGA不包含RAM 的功能，所以外加一個 RAM 才能形成 computer
'''
