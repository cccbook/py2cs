from turingMachine import TuringMachine

# Construct a TM for the language L = {0n1n2n} where n≥1
# https://www.cs.odu.edu/~zeil/cs390/latest/Public/turing-jflap/index.html
# 3.1 $0^n1^n2^n$ 改為 anbncn, 0=a,1=b,2=c,epsilon=_
tm = TuringMachine(
    ['0','1','2','3','4','5'], # 狀態集合 Q
    ['a','b','c'],             # 輸入字母表
    ['a','b','c','X','_'],     # 磁帶字母表
    {
        '0,a':'1,_,R', # 狀態 0 遇到 a 轉到 1，寫入 _, 然後向右方移動
        '1,a':'1,a,R', # 接下來都是類似的解讀方法 ...
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
    '0',   # 起始狀態為 0
    ['3'], # 接受狀態集合是 3
    []     # 沒有拒絕狀態
    )

print(':', tm.run(''))
print('ab:', tm.run('ab'))
print('abtt:', tm.run('abtt'))
print('abc:', tm.run('abc'))
print('aabbc:', tm.run('aabbc'))
print('aabbcc:', tm.run('aabbcc'))
