import math
from collections import defaultdict

class MaxEnt:
    def __init__(self):
        self.weights = defaultdict(float)  # 存儲每個特徵對應的權重
        self.labels = set()  # 用來存儲所有可能的標籤
    
    def extract_features(self, sentence, index):
        word = sentence[index][0]  # 當前單詞
        prev_word = sentence[index - 1][0] if index > 0 else "<START>"  # 上一個單詞
        return {
            'word': word,
            'prev_word': prev_word,
            'word_length': len(word),
            'ends_with_ing': word.endswith('ing')
        }

    def get_feature_vector(self, features, label):
        # 將每個特徵與標籤結合
        return [f"{key}={value},label={label}" for key, value in features.items()]

    def predict(self, features):
        scores = defaultdict(float)

        # 計算每個標籤的得分
        for label in self.labels:
            feature_vector = self.get_feature_vector(features, label)
            for feature in feature_vector:
                scores[label] += self.weights[feature]

        # 找出最大得分的標籤
        max_label = max(scores, key=scores.get)
        return max_label

    def train(self, training_data, iterations=100):
        for features, label in training_data:
            self.labels.add(label)  # 訓練過程中收集所有的標籤
        
        for _ in range(iterations):
            for features, label in training_data:
                # 預測當前特徵的標籤
                predicted_label = self.predict(features)

                # 如果預測錯誤，更新權重
                if predicted_label != label:
                    correct_feature_vector = self.get_feature_vector(features, label)
                    predicted_feature_vector = self.get_feature_vector(features, predicted_label)

                    # 增加正確標籤的權重，減少錯誤標籤的權重
                    for feature in correct_feature_vector:
                        self.weights[feature] += 1
                    for feature in predicted_feature_vector:
                        self.weights[feature] -= 1

    def tag_sentence(self, sentence):
        tagged_sentence = []
        for index in range(len(sentence)):
            features = self.extract_features(sentence, index)
            predicted_label = self.predict(features)
            tagged_sentence.append((sentence[index][0], predicted_label))
        return tagged_sentence

# 訓練數據
training_data = [
    ({"word": "The", "prev_word": "<START>", "word_length": 3, "ends_with_ing": False}, "DT"),
    ({"word": "cat", "prev_word": "The", "word_length": 3, "ends_with_ing": False}, "NN"),
    ({"word": "sat", "prev_word": "cat", "word_length": 3, "ends_with_ing": False}, "VBD"),
    ({"word": "on", "prev_word": "sat", "word_length": 2, "ends_with_ing": False}, "IN"),
    ({"word": "the", "prev_word": "on", "word_length": 3, "ends_with_ing": False}, "DT"),
    ({"word": "mat", "prev_word": "the", "word_length": 3, "ends_with_ing": False}, "NN"),
]

# 初始化模型並訓練
model = MaxEnt()
model.train(training_data)

# 測試句子
test_sentence = [("The", None), ("cat", None), ("runs", None)]
tagged_output = model.tag_sentence(test_sentence)

print("Tagged output:", tagged_output)
