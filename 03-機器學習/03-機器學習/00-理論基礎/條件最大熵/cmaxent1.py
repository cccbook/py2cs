import nltk
from nltk.corpus import treebank
from nltk.classify import MaxentClassifier
from sklearn.model_selection import train_test_split

# 下載 nltk 標註數據集
nltk.download('treebank')

# 加載詞性標註數據
data = treebank.tagged_sents()

# 定義特徵函數
def extract_features(sentence, index):
    word = sentence[index][0]  # 當前單詞
    prev_word = sentence[index-1][0] if index > 0 else "<START>"  # 上一個單詞
    return {
        'word': word,
        'prev_word': prev_word,
        'word_length': len(word),
        'ends_with_ing': word.endswith('ing'),
    }

# 準備數據
featuresets = []
labels = []

for sentence in data:
    for index in range(len(sentence)):
        features = extract_features(sentence, index)
        label = sentence[index][1]  # 詞性標籤
        featuresets.append((features, label))

# 切分數據集為訓練集和測試集
train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=42)

# 訓練最大熵模型
classifier = MaxentClassifier.train(train_set)

# 測試模型
def tag_sentence(sentence):
    tagged_sentence = []
    for index in range(len(sentence)):
        features = extract_features(sentence, index)
        label = classifier.classify(features)
        tagged_sentence.append((sentence[index][0], label))
    return tagged_sentence

# 測試範例句子
test_sentence = [("The", None), ("cat", None), ("runs", None)]
tagged_output = tag_sentence(test_sentence)

print("Tagged output:", tagged_output)
