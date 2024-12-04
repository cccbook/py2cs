import rsa

# 生成 RSA 密鑰對
def generate_rsa_keys():
    public_key, private_key = rsa.newkeys(2048)
    return private_key, public_key

# 使用公開密鑰加密訊息
def encrypt_message(public_key, message):
    # 將訊息轉換為字節並加密
    encrypted_message = rsa.encrypt(message.encode('utf-8'), public_key)
    return encrypted_message

# 使用私密密鑰解密訊息
def decrypt_message(private_key, encrypted_message):
    # 解密訊息
    decrypted_message = rsa.decrypt(encrypted_message, private_key).decode('utf-8')
    return decrypted_message

# 主程式
def main():
    # 生成密鑰對
    private_key, public_key = generate_rsa_keys()

    # 顯示公開和私密密鑰（以方便示範）
    print("公開密鑰:", public_key)
    print("私密密鑰:", private_key)

    # 要加密的訊息
    message = "Hello, this is a secret message!"
    print("\n原始訊息:", message)

    # 加密訊息
    encrypted_message = encrypt_message(public_key, message)
    print("\n加密後的訊息:", encrypted_message)

    # 解密訊息
    decrypted_message = decrypt_message(private_key, encrypted_message)
    print("\n解密後的訊息:", decrypted_message)

if __name__ == "__main__":
    main()
