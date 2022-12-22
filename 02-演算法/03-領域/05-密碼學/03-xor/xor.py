def encrypt(msg, key):
    r = []
    for i in range(len(msg)):
        k = key[i%len(key)]
        r.append(msg[i] ^ k)
    return bytearray(r)

def decrypt(msg, key):
    return encrypt(msg, key)

key = [17, 3, 15]
msg = b"Attack at dawn"
print('msg=', msg)
cipher = encrypt(msg, key)
print('cipher=', cipher)
plain = decrypt(cipher, key)
print('plain=', plain)
