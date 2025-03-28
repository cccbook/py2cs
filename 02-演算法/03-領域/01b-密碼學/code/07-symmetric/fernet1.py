from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher_suite = Fernet(key)
plainText = "Hello 你好！"
cipherBytes = cipher_suite.encrypt(str.encode(plainText))
plainBytes = cipher_suite.decrypt(cipherBytes)
plainText2 = plainBytes.decode()
print(plainText2)
