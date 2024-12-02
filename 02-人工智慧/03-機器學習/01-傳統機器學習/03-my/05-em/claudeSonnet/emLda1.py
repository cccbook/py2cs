import numpy as np
from scipy.special import digamma
from collections import Counter

def initialize_parameters(docs, vocab_size, num_topics):
    doc_topic_counts = np.random.rand(len(docs), num_topics)
    topic_word_counts = np.random.rand(num_topics, vocab_size)
    return doc_topic_counts, topic_word_counts

def e_step(doc, doc_topic_counts, topic_word_counts):
    doc_topic_dist = doc_topic_counts[doc.id] + 1
    topic_word_dist = topic_word_counts[:, [word for word in doc.words]] + 1
    topic_dist = doc_topic_dist * topic_word_dist / topic_word_dist.sum(axis=1)[:, np.newaxis]
    return topic_dist / topic_dist.sum(axis=0)

def m_step(docs, topic_assignments):
    doc_topic_counts = np.array([ta.sum(axis=1) for ta in topic_assignments])
    topic_word_counts = sum(ta[:, np.newaxis] * doc.one_hot for doc, ta in zip(docs, topic_assignments))
    return doc_topic_counts, topic_word_counts

def lda_em(docs, vocab_size, num_topics, num_iterations=50):
    doc_topic_counts, topic_word_counts = initialize_parameters(docs, vocab_size, num_topics)
    
    for _ in range(num_iterations):
        # E-step
        topic_assignments = [e_step(doc, doc_topic_counts, topic_word_counts) for doc in docs]
        
        # M-step
        doc_topic_counts, topic_word_counts = m_step(docs, topic_assignments)
    
    return doc_topic_counts, topic_word_counts

class Document:
    def __init__(self, id, words, vocab):
        self.id = id
        self.words = words
        self.one_hot = np.zeros((len(words), len(vocab)))
        for i, word in enumerate(words):
            self.one_hot[i, vocab[word]] = 1

# Example usage
vocab = {"apple": 0, "banana": 1, "cherry": 2, "date": 3, "elderberry": 4, "fig": 5}
docs = [
    Document(0, ["apple", "banana", "cherry"], vocab),
    Document(1, ["banana", "cherry", "date"], vocab),
    Document(2, ["cherry", "date", "elderberry"], vocab),
    Document(3, ["date", "elderberry", "fig"], vocab),
    Document(4, ["elderberry", "fig", "apple"], vocab),
]

num_topics = 2
doc_topic_counts, topic_word_counts = lda_em(docs, len(vocab), num_topics)

print("Document-Topic Distribution:")
print(doc_topic_counts)
print("\nTopic-Word Distribution:")
print(topic_word_counts)

# Print top words for each topic
for topic in range(num_topics):
    top_words = sorted(range(len(vocab)), key=lambda i: topic_word_counts[topic, i], reverse=True)[:3]
    print(f"\nTop words for Topic {topic}:")
    print(", ".join([list(vocab.keys())[i] for i in top_words]))