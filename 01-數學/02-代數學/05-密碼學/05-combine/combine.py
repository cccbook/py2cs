INT_BITS = 8
 
def leftRotate(n, d, bits):
    return (2**bits-1)&((n << d)|(n >> (bits - d)))
 
def rightRotate(n, d, bits):
    return (2**bits-1)&((n >> d)|(n << (bits - d)))

def encrypt(msg, rotKey, xorKey):
    r = []
    for i in range(len(msg)):
        m = msg[i]
        m = leftRotate(m, rotKey, 8)
        m = m ^ xorKey
        r.append(m)
    return bytearray(r)

def decrypt(msg, rotKey, xorKey):
    r = []
    for i in range(len(msg)):
        m = msg[i]
        m = m^xorKey
        m = rightRotate(m, rotKey, 8)
        r.append(m)
    return bytearray(r)

rotKey = b"\x03"[0]
xorKey = b"\x7e"[0]
msg = b"Attack at dawn"
print('msg=', msg)
cipher = encrypt(msg, rotKey, xorKey)
print('cipher=', cipher)
plain = decrypt(cipher, rotKey, xorKey)
print('plain=', plain)
