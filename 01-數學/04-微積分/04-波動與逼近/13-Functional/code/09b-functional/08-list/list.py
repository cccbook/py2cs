from pair import *

def list_new(*args):
    return (None if len(args) == 0 else
            pair(args[0], list_new(*args[1:]))
    )

def list_str(items):
    return (None if items is None else 
            f'[{head(items)},{list_str(tail(items))}]'
    )

def list_print(items):
    print(list_str(items))

def list_ref(items, n):
    return (head(items) if n == 0 else
            list_ref(tail(items), n - 1)
    )

def length(items):
    return (0 if items is None else
            1 + length(tail(items))
    )

def append(list1, list2):
    return (list2 if list1 is None else
            pair(head(list1), append(tail(list1), list2))
    )

def map(fun, items):
    return (None if items is None else
            pair(fun(head(items)), map(fun, tail(items)))
    )

def foreach(fun, items):
    return (fun(None) if items is None else
           pair(fun(head(items)), foreach(fun, tail(items)))
    )
