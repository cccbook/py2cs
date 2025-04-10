def encrypt(msg, key):
    r = []
    for i in range(len(msg)):
        k = key[i%len(key)]
        r.append((msg[i]+k+256)%256)
    return bytearray(r)

def decrypt(msg, key):
    r = []
    for i in range(len(msg)):
        k = key[i%len(key)]
        r.append((msg[i]-k+256)%256)
    return bytearray(r)

key = [17, 3, 15]
msg = b"Attack at dawn"
print('msg=', msg)
cipher = encrypt(msg, key)
print('cipher=', cipher)
plain = decrypt(cipher, key)
print('plain=', plain)
