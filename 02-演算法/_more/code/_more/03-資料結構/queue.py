class Queue:
    def __init__(self):
        self.a = []
    
    def enqueue(self, o):
      self.a.append(o)

    def dequeue(self):
      return self.a.pop(0)

    def __str__(self):
        return f"{self.a}"

q = Queue()
q.enqueue(3)
print(f"q={q}")
q.enqueue(5)
print(f"q={q}")
q.enqueue(2)
print(f"q={q}")
t1 = q.dequeue()
print(f"t1={t1}")
print(f"q={q}")
