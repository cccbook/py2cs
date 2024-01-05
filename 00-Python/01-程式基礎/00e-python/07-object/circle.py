class Circle:
    def __init__(self, r):
        self.r = r
    def area(self):
        return 3.14*self.r*self.r

c = Circle(2)
print('c.r=', c.r, 'c.area()=', c.area())
