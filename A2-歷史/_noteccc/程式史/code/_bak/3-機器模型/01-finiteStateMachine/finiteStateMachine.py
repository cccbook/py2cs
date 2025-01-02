import json

class FiniteStateMachine:
    def __init__(self, start, finals, actionMap):
        self.actionMap = actionMap
        self.start = start
        self.finals = finals
    def accept(self, s):
        state = self.start
        i = 0
        while True:
            if i >= len(s): break
            state = self.actionMap.get(f'{state},{s[i]}')
            if state is None: return False
            i += 1
        return state in self.finals
    def __str__(self):
        return json.dumps({'start':self.start,'finals':self.finals,'actionMap':self.actionMap}, indent=2)

