# ChatGPT: https://chatgpt.com/share/66f20fcb-c5fc-8012-932b-92e779254af0

from bitarray import bitarray
import mmh3  # MurmurHash3

class BloomFilter:
    def __init__(self, size, hash_count):
        # 初始化位元陣列，大小為 size，並設定哈希函數的數量
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
    
    def add(self, item):
        # 將 item 經過多個哈希函數後，標記位元陣列
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            self.bit_array[digest] = 1
    
    def check(self, item):
        # 檢查 item 是否存在
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            if self.bit_array[digest] == 0:
                return False
        return True

# 測試 Bloom Filter
bloom = BloomFilter(100, 5)  # 位元陣列大小為100，使用5個哈希函數

# 新增元素到 Bloom Filter
bloom.add("apple")
bloom.add("banana")
bloom.add("orange")

# 檢查元素是否在 Bloom Filter 中
result_apple = bloom.check("apple")
result_grape = bloom.check("grape")

(result_apple, result_grape)  # 顯示 "apple" 應該存在, "grape" 應該不存在
