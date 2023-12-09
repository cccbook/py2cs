class SetA:
    def contain(self, e):
        return not e.contain(e)

A = SetA()

print('A.contain(A)=', A.contain(A))
