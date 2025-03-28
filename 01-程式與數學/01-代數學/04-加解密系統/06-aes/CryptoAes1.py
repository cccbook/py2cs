# https://pycryptodome.readthedocs.io/en/latest/src/examples.html
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

data = b'secret data'

key = get_random_bytes(16)
cipher1 = AES.new(key, AES.MODE_EAX)
ciphertext, tag = cipher1.encrypt_and_digest(data)

cipher2 = AES.new(key, AES.MODE_EAX, nonce)
data2 = cipher2.decrypt_and_verify(ciphertext, tag)
print('data2=', data2)
