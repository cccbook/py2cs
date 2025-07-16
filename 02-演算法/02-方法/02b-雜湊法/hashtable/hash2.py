def hash(s):
    h = 5381
    i = len(s)-1
    while i>=0 :
        h = (h * 33) ^ ord(s[i])
        i -= 1 
    return int(h)

print('hash(hello)=', hash('hello'))
print('hash(hello!)=', hash('hello!'))
print('hash(hello world !)=', hash('hello world !'))