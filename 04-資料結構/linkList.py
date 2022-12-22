class Node():
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList():
    def __init__(self):
        self.head = None
        self.tail = self.head

    def append(self, data):
        new_node = Node(data)
        if self.head == None:
            self.head = new_node
            self.tail = self.head
        else:
            self.tail.next = new_node
            self.tail = new_node

    def print(self):
        if self.head == None:
            print("None")
        else:
            current_node = self.head
            while current_node!= None:
                print(current_node.data, end= ' ')
                current_node = current_node.next
            print("None")

list1 = LinkedList()
list1.print()

list1.append("a")
list1.append(1)
list1.append(3.14159)
list1.append(["b","c"])
list1.print()


