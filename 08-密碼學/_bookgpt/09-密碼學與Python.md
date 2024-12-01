**第 9 章 密碼學與 Python**

---

### **9.1 Python 密碼學庫概覽**

#### **1. PyCryptodome**
- **功能**：
  - 支持對稱加密（AES、DES 等）。
  - 支持非對稱加密（RSA、ECC 等）。
  - 支持散列函數（SHA、MD5 等）。
- **優點**：簡單易用，適合入門。
- **安裝**：
  ```bash
  pip install pycryptodome
  ```

#### **2. Cryptography**
- **功能**：
  - 支持高級密碼學功能。
  - 提供 Fernet 加密（對稱加密方案）。
- **優點**：安全性高，社群活躍。
- **安裝**：
  ```bash
  pip install cryptography
  ```

#### **3. passlib**
- **功能**：
  - 用於處理密碼散列（如 bcrypt）。
  - 支持多種算法。
- **優點**：適合密碼存儲的應用。

---

### **9.2 實現加密通信應用**

#### **應用場景**
- 安全地在兩個節點之間傳輸數據。
- 防止第三方截獲或篡改數據。

#### **Python 示例：加密通信**
以下示例展示如何使用 AES 實現簡單的加密通信。

```python
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

# 發送端
def send_message(message, key):
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(pad(message.encode(), AES.block_size))
    return cipher.nonce, ciphertext, tag

# 接收端
def receive_message(nonce, ciphertext, tag, key):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = unpad(cipher.decrypt_and_verify(ciphertext, tag), AES.block_size)
    return plaintext.decode()

# 示例
key = get_random_bytes(16)  # 16 字節密鑰
nonce, ciphertext, tag = send_message("Hello, secure world!", key)
print(f"Encrypted message: {ciphertext}")

plaintext = receive_message(nonce, ciphertext, tag, key)
print(f"Decrypted message: {plaintext}")
```

---

### **9.3 密碼學在區塊鏈中的應用**

#### **1. 哈希算法**
- 區塊鏈中每個區塊的哈希值用於確保數據完整性。
- 常用算法：SHA-256。

#### **2. 非對稱加密**
- 用於數字簽章和交易驗證。
- 比特幣使用 ECDSA（橢圓曲線數字簽章算法）。

#### **3. Merkle 樹**
- 使用哈希樹結構提高數據驗證效率。

#### **Python 示例：簡化版區塊哈希**
```python
import hashlib
import json

def create_block(index, data, prev_hash):
    block = {
        "index": index,
        "data": data,
        "prev_hash": prev_hash,
    }
    block["hash"] = hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()
    return block

# 創建區塊鏈
blockchain = []
genesis_block = create_block(0, "Genesis Block", "0")
blockchain.append(genesis_block)

new_block = create_block(1, "Transaction Data", genesis_block["hash"])
blockchain.append(new_block)

for block in blockchain:
    print(block)
```

---

### **9.4 安全程式設計原則**

#### **1. 最小權限原則**
- 確保程式僅擁有執行必要功能所需的權限。
- 避免過度授權引發安全問題。

#### **2. 輸入驗證**
- 防止 SQL 注入、跨站腳本攻擊（XSS）等漏洞。
- 避免不信任的輸入直接傳遞到密碼學函數。

#### **3. 加密敏感數據**
- 明文存儲敏感信息（如密碼）是常見錯誤。
- 使用強散列算法存儲密碼。

#### **4. 密鑰管理**
- 將密鑰存儲在安全位置，例如硬體安全模組（HSM）。
- 避免硬編碼密鑰到程式碼中。

#### **Python 示例：安全密碼存儲**
```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 加密密碼
def hash_password(password):
    return pwd_context.hash(password)

# 驗證密碼
def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)

# 示例
hashed_password = hash_password("securepassword123")
print(f"Hashed password: {hashed_password}")

is_valid = verify_password("securepassword123", hashed_password)
print(f"Password valid: {is_valid}")
```

---

**本章小結**：
- 本章介紹了多個 Python 密碼學庫及其應用場景。
- 實現了加密通信和區塊鏈應用的基本示例。
- 強調了安全程式設計的基本原則，幫助讀者避免常見的安全漏洞。

下一章將探討密碼學未來的挑戰與發展，包括量子計算對密碼學的影響和後量子密碼學的最新進展。