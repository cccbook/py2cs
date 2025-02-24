import sys
sys.path.append('../11-bigPrimeRsa/')
import rsa
import bigPrime as bp

e, d, N = rsa.genKeyPair(200) # Alice generate e d N
print('e=', e)            # Alice publish e N
print('d=', d)
print('N=', N)
k1 = bp.randomBigInt(100)
c = rsa.encrypt(e, N, k1) # Bob:e,N = c => Alice
k2 = rsa.decrypt(d, N, c) # Alice:d,N : decode c into k2
print('k1=', k1)
print('c=', c)
print('k2=', k2)
assert k1==k2

sys.path.append('../06-aes/')
import aes

key = f"{k1}" # Alice, Bob: both know key.
message = b'a secret message, attack at dawn, 4:50'
print('message=', message)
# Alice, Bob use key to pass message secretly.
cipherText = aes.encrypt(key, message)
print('cipherText=', cipherText)
plainText = aes.decrypt(key, cipherText)
print('plainText=', plainText)
assert message == plainText
