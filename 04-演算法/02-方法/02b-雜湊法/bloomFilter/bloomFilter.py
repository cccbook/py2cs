# ChatGPT: https://chatgpt.com/share/66f20fcb-c5fc-8012-932b-92e779254af0

import hashlib

class SimpleBloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size  # 使用列表來模擬位元陣列
    
    def _hashes(self, item):
        # 使用不同哈希方法來模擬多個哈希函數
        result = []
        for i in range(self.hash_count):
            hash_value = int(hashlib.md5((item + str(i)).encode()).hexdigest(), 16)
            result.append(hash_value % self.size)
        return result
    
    def add(self, item):
        # 將 item 經過多個哈希函數後，標記位元陣列
        for hash_value in self._hashes(item):
            self.bit_array[hash_value] = 1
    
    def check(self, item):
        # 檢查 item 是否存在
        for hash_value in self._hashes(item):
            if self.bit_array[hash_value] == 0:
                return False
        return True

# 測試 Simple Bloom Filter
bloom = SimpleBloomFilter(100, 5)  # 位元陣列大小為100，使用5個哈希函數

# 新增元素到 Bloom Filter
bloom.add("apple")
bloom.add("banana")
bloom.add("orange")

# 檢查元素是否在 Bloom Filter 中
result_apple = bloom.check("apple")
result_grape = bloom.check("grape")

print('apple:', result_apple) # "apple" 應該存在,
print('grape:', result_grape) # "grape" 應該不存在
