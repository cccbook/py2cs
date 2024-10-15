import numpy as np
import nltk
from nltk.corpus import gutenberg
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import OneHotEncoder

# 確保已下載 NLTK Gutenberg 語料庫
nltk.download('punkt')
nltk.download('gutenberg')

class DBN:
    def __init__(self, n_hidden_layers, n_units_per_layer):
        self.n_hidden_layers = n_hidden_layers
        self.n_units_per_layer = n_units_per_layer
        self.rbms = []
    
    def fit(self, X):
        for i in range(self.n_hidden_layers):
            rbm = BernoulliRBM(n_components=self.n_units_per_layer[i], learning_rate=0.01, n_iter=10)
            if i == 0:
                rbm.fit(X)
                X = rbm.transform(X)
            else:
                rbm.fit(X)
                X = rbm.transform(X)
            self.rbms.append(rbm)
    
    def transform(self, X):
        for rbm in self.rbms:
            X = rbm.transform(X)
        return X

# 將文本數據轉換為數值格式
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    return words

# 從 Gutenberg 語料庫中加載文本
def load_corpus():
    text = gutenberg.raw('carroll-alice.txt')
    return preprocess_text(text)

# 將文本轉換為特徵矩陣
def create_feature_matrix(words):
    word_counts = nltk.FreqDist(words)
    vocabulary = list(word_counts.keys())
    
    # 建立 OneHotEncoder 來編碼單詞
    encoder = OneHotEncoder(sparse=False)
    encoded_matrix = encoder.fit_transform(np.array(vocabulary).reshape(-1, 1))
    
    features = []
    for word in words:
        if word in vocabulary:
            features.append(encoded_matrix[vocabulary.index(word)])
    
    return np.array(features), vocabulary

# 生成接龍文本
def generate_text(dbn, vocabulary, start_word, num_words=10):
    generated_text = [start_word]
    current_word = start_word

    for _ in range(num_words - 1):
        # 將當前單詞轉換為特徵向量
        if current_word in vocabulary:
            index = vocabulary.index(current_word)
            input_vector = np.zeros((1, len(vocabulary)))
            input_vector[0][index] = 1
            
            # 將輸入向量變換
            transformed_vector = dbn.transform(input_vector)
            # 從變換的結果選擇下個單詞（這裡使用簡單的隨機選擇）
            next_index = np.random.choice(len(vocabulary))
            current_word = vocabulary[next_index]
            generated_text.append(current_word)
        else:
            break  # 當前單詞不在詞彙中時結束生成

    return ' '.join(generated_text)

# 主程序
if __name__ == "__main__":
    words = load_corpus()
    X, vocabulary = create_feature_matrix(words)

    # 設定 DBN 結構
    n_hidden_layers = 2
    n_units_per_layer = [100, 50]

    # 建立 DBN 模型
    dbn = DBN(n_hidden_layers, n_units_per_layer)

    # 訓練模型
    dbn.fit(X)

    # 使用訓練好的模型生成接龍文本
    start_word = np.random.choice(vocabulary)  # 隨機選擇起始單詞
    generated_text = generate_text(dbn, vocabulary, start_word, num_words=10)
    
    print("Generated Text:\n", generated_text)
