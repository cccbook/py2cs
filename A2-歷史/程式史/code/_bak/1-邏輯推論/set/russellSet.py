class Set:
    def has(self, e):
        pass

class SetA(Set):
    def has(self, e):
        return not e.has(e)

A = SetA()

print('A.has(A)=', A.has(A))
