INT_BITS = 8
 
def leftRotate(n, d, bits):
    return (2**bits-1)&((n << d)|(n >> (bits - d)))
 
def rightRotate(n, d, bits):
    return (2**bits-1)&((n >> d)|(n << (bits - d)))

def encrypt(msg, key):
    r = []
    for i in range(len(msg)):
        r.append(leftRotate(msg[i], key, 8))
    print('r=', r)
    return bytearray(r)

def decrypt(msg, key):
    r = []
    for i in range(len(msg)):
        r.append(rightRotate(msg[i], key, 8))
    return bytearray(r)

key = b"\x03"[0]
msg = b"Attack at dawn"
print('msg=', msg)
cipher = encrypt(msg, key)
print('cipher=', cipher)
plain = decrypt(cipher, key)
print('plain=', plain)
