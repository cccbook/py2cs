from finiteStateMachine import FiniteStateMachine

fsm = FiniteStateMachine('s0', ['s0'], {
    's0,a':'s0'
})
print(fsm)
print('aaa:', fsm.accept('aaa'))
print('aab:', fsm.accept('aab'))
print('aaaaaaa:', fsm.accept('aaaaaaa'))
