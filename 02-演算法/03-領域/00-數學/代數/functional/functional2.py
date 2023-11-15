class Function:
    def __init__(self, f):
        self.f = f
    def __neg__(self):
        return Function(lambda x:-self.f(x))
    def __inv__(self):
        return Function(lambda x:1/self.f(x))
    def __add__(self, b):
        return Function(lambda x:self.f(x)+b.f(x))
    def __mul__(self, b):
        return Function(lambda x:self.f(x)*b.f(x))
    def __sub__(self, b):
        return Function(lambda x:self.f(x)-b.f(x))
    def __truediv__(self, b):
        return Function(lambda x:self.f(x)/b.f(x))
    def __str__(self):
        return str(self.f)
    def compose(self, b):
        return Function(lambda x:self.f(b.f(x)))
    def __call__(self, x):
        return self.f(x)

a = Function(lambda x:x); b = Function(lambda x:x**2)
print('-a(2)=', (-a)(2))
print('(a+b)(2)=', (a+b)(2))
print('(a*b)(2)=', (a*b)(2))
print('(b/a)(2)=', (b/a)(2))
