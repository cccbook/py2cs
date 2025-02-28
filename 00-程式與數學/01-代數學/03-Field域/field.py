class Field:
    def sub(self, b):
        return self.add(b.neg())
    def div(self, b):
        return self.mul(b.inv())

class Number(Field):
    def __init__(self, f):
        self.v = f
    def neg(self):
        return Number(-self.v)
    def inv(self):
        return Number(1/self.v)
    def add(self, b):
        return Number(self.v+b.v)
    def mul(self, b):
        return Number(self.v*b.v)
    def __str__(self):
        return str(self.v)

a = Number(3); b = Number(2)
print('a.add(b)=', a.add(b))
print('a.sub(b)=', a.sub(b))
print('a.mul(b)=', a.mul(b))
print('a.div(b)=', a.div(b))

a = Number(3+2j); b = Number(2+3j)
print('a.add(b)=', a.add(b))
print('a.sub(b)=', a.sub(b))
print('a.mul(b)=', a.mul(b))
print('a.div(b)=', a.div(b))


