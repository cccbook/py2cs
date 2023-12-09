# https://zh.wikipedia.org/wiki/%E5%9B%BE%E7%81%B5%E6%9C%BA
# https://www.cs.odu.edu/~zeil/cs390/latest/Public/turing-jflap/index.html
class TuringMachine:
    def __init__(self, states, ichars, ochars, actionMap, start, accepts, rejects):
        self.states = states
        self.ichars = ichars
        self.ochars = ochars
        self.actionMap = actionMap
        self.start = start
        self.accepts = accepts
        self.rejects = rejects

    def action(self, state, ichar):
        line = self.actionMap.get(f'{state},{ichar}')
        if line == None: return None
        return line.split(',')

    # 參考 -- https://zh.wikipedia.org/wiki/%E5%9B%BE%E7%81%B5%E6%9C%BA#%E5%9B%BE%E7%81%B5%E6%9C%BA%E7%9A%84%E6%AD%A3%E5%BC%8F%E5%AE%9A%E4%B9%89
    def run(self, tape):
        self.tape = list(tape+'________') # 开始的时候将输入符号串，其他格子保持空白
        state = self.start # 機器一開始處於起始狀態
        i = 0 # 读写头指向第0号格子
        while True:
            ichar = self.tape[i]
            acts = self.action(state, ichar)
            if acts == None: return False
            state, ochar, move = acts # 執行動作
            self.tape[i] = ochar # 改寫這一格符號
            if move == 'L':
                if i > 0: i -= 1 # 向左移動 (但不能越過磁帶開頭)
            elif move == 'R': i += 1 # 向右移動
            elif move == '-': pass # 不移動
            else: raise Exception(f'move = {move} not allowed!')
            if state in self.accepts:
                return True
            if state in self.rejects:
                return False

if __name__=="__main__":
    # Construct a TM for the language L = {0n1n2n} where n≥1
    # https://www.cs.odu.edu/~zeil/cs390/latest/Public/turing-jflap/index.html
    # 3.1 $0^n1^n2^n$ 改為 anbncn, 0=a,1=b,2=c,epsilon=_
    tm = TuringMachine(
        ['0','1','2','3','4','5'],
        ['a','b','c'],
        ['a','b','c','X','_'],
        {
            '0,a':'1,_,R',
            '1,a':'1,a,R',
            '1,x':'1,x,R',
            '1,b':'2,x,R',
            '2,x':'2,x,R',
            '2,b':'2,b,R',
            '2,c':'5,x,L',
            '5,x':'5,x,L',
            '5,a':'5,a,L',
            '5,b':'5,b,L',
            '5,_':'0,_,R',
            '0,x':'4,x,R',
            '0,_':'3,_,L',
            '4,x':'4,x,R',
            '4,_':'3,_,L',
        },
        '0',
        ['3'],
        []
        )

    print(':', tm.run(''))
    print('ab:', tm.run('ab'))
    print('abtt:', tm.run('abtt'))
    print('abc:', tm.run('abc'))
    print('aabbc:', tm.run('aabbc'))
    print('aabbcc:', tm.run('aabbcc'))
