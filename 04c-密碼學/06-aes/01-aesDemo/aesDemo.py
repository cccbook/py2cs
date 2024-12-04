from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os

# 生成密鑰和初始化向量
def generate_key_and_iv():
    key = os.urandom(32)  # 32 bytes for AES-256
    iv = os.urandom(16)   # 16 bytes for AES block size
    return key, iv

# 加密函數
def encrypt(plaintext, key, iv):
    # 使用 PKCS7 填充
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plaintext.encode()) + padder.finalize()

    # 設定 AES 加密模式 (CBC模式)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # 執行加密
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return ciphertext

# 解密函數
def decrypt(ciphertext, key, iv):
    # 設定 AES 解密模式 (CBC模式)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    # 解密
    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()

    # 去除填充
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(decrypted_data) + unpadder.finalize()
    return plaintext.decode()

# 測試加解密過程
if __name__ == "__main__":
    key, iv = generate_key_and_iv()  # 生成密鑰和初始化向量
    message = "這是秘密訊息"  # 要加密的訊息
    
    print(f"原文: {message}")
    
    # 加密
    encrypted = encrypt(message, key, iv)
    print(f"加密後的訊息: {encrypted.hex()}")
    
    # 解密
    decrypted = decrypt(encrypted, key, iv)
    print(f"解密後的訊息: {decrypted}")
