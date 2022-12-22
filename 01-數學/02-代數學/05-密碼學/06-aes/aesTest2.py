import aes

key = "8913914313143121"
message = b'a secret message'
print('message=', message)
cipherText = aes.encrypt(key, message)
print('cipherText=', cipherText)
plainText = aes.decrypt(key, cipherText)
print('plainText=', plainText)

