from aes import AES

aes = AES(b'\x73' * 16)
message = b'a secret message'
print('message=', message)
cipherText = aes.encrypt_block(message)
print('cipherText=', cipherText)
plainText = aes.decrypt_block(cipherText)
print('plainText=', plainText)

