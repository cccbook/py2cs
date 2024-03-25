import torch

class MyEmbedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 初始化嵌入矩陣（權重）
        self.weight = torch.randn(num_embeddings, embedding_dim)

    def forward(self, input):
        # 輸入是一個張量，每個元素是一個索引，指示要從嵌入矩陣中提取哪些嵌入向量
        # 對於每個索引，從嵌入矩陣中提取相應的嵌入向量
        return self.weight[input]

# 測試自行實作的Embedding層
if __name__ == "__main__":
    # 建立一個輸入張量（假設是一個包含5個單詞的句子，每個單詞的索引都在範圍內）
    input_indices = torch.tensor([1, 3, 0, 4, 2])

    # 建立自行實作的Embedding層（單詞表大小為6，每個單詞嵌入維度為3）
    embedding_layer = MyEmbedding(num_embeddings=6, embedding_dim=3)

    # 將輸入張量通過自行實作的Embedding層
    output = embedding_layer.forward(input_indices)

    print("輸入張量（單詞索引）:")
    print(input_indices)
    print("自行實作的Embedding後的輸出:")
    print(output)
