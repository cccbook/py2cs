def hashCode(s):
    hash = 0 
    if len(s) == 0: return hash
    for i in range(len(s)):
        c     = s[i]
        hash  = ((hash << 5) - hash) + ord(c) # hash = hash*31 + chr = (hash*32-hash) + c
        hash  = int(hash)
    return hash

print('hashCode(hello)=', hashCode('hello'))
print('hashCode(hello!)=', hashCode('hello!'))
print('hashCode(hello world !)=', hashCode('hello world !'))
