import hashlib

def hash(text):
    m = hashlib.sha256()
    m.update(text.encode('utf-8'))
    return m.hexdigest()

print('hash(hello)=', hash('hello'))
print('hash(hello!)=', hash('hello!'))
print('hash(hello world !)=', hash('hello world !'))
