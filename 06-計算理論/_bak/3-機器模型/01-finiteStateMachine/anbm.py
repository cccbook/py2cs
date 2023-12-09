# 參考 https://zh.wikipedia.org/wiki/%E6%9C%89%E9%99%90%E7%8A%B6%E6%80%81%E6%9C%BA#/media/File:DFAexample.svg
from finiteStateMachine import FiniteStateMachine

fsm = FiniteStateMachine('s1', ['s2'], {
    's1,a':'s1',
    's1,b':'s2',
    's2,b':'s2',
})
print(fsm)
print('a:', fsm.accept('a'))
print('aab:', fsm.accept('aab'))
print('aabaa:', fsm.accept('aabaa'))
print('aabbb:', fsm.accept('aabbb'))
