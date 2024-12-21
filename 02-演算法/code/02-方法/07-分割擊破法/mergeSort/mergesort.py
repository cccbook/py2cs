import math

def sort(array):
    length = len(array)
    mid    = int(length/2)
    left   = array[:mid]
    right  = array[mid:]
    if length == 1: return array
    return merge(sort(left), sort(right))


def merge(left, right):
    result = [] 
    while len(left)>0 or len(right)>0:
        if len(left)>0 and len(right)>0:
            result.append(left.pop(0)) if left[0]<right[0] else result.append(right.pop(0)) 
        elif len(left)>0:
            result.append(left.pop(0)) 
        else:
            result.append(right.pop(0)) 
    return result 



