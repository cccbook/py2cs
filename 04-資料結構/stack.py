class Stack:
    def __init__(self):
        self.a = []

    def push(self, o):
        self.a.append(o)

    def pop(self):
        return self.a.pop()

    def __str__(self):
        return f'{self.a}'

s = Stack()
s.push(3)
s.push(5)
print("s=", s)
s.push(2)
print("s=", s)
t1 = s.pop()
print("t1=", t1)
print("s=", s)

