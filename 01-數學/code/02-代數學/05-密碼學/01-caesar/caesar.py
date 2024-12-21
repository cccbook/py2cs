def encrypt(msg, k):
    r = []
    for b in msg:
        r.append((b+k+256)%256)
    return bytearray(r)

def decrypt(msg, k):
    return encrypt(msg, -k)

key = 17
msg = b"Attack at dawn"
print('msg=', msg)
cipher = encrypt(msg, key)
print('cipher=', cipher)
plain = decrypt(cipher, key)
print('plain=', plain)
