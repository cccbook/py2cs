# 參考 https://zh.wikipedia.org/wiki/%E6%9C%89%E9%99%90%E7%8A%B6%E6%80%81%E6%9C%BA#/media/File:DFAexample.svg
from finiteStateMachine import FiniteStateMachine

fsm = FiniteStateMachine('s1', ['s1'], {
    's1,0':'s2',
    's1,1':'s1',
    's2,0':'s1',
    's2,1':'s2'
})
print(fsm)
print('010:', fsm.accept('010'))
print('101:', fsm.accept('101'))
print('10100:', fsm.accept('10100'))
print('1010010:', fsm.accept('1010010'))