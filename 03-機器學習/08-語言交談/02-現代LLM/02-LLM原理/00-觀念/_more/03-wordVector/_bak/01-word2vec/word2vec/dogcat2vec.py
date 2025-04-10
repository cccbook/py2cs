# 來源 -- https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
# Python program to generate word vectors using Word2Vec 
  
# importing all necessary modules
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
  
warnings.filterwarnings(action = 'ignore') 
  
import gensim
from gensim.models import Word2Vec 
  
#  Reads ‘alice.txt’ file 
# sample = open("alice.txt", "r") 
sample = open("dogcat.txt", "r") 
s = sample.read() 
  
# Replaces escape character with space 
f = s.replace("\n", " ") 
  
data = [] 
  
# iterate through each sentence in the file 
for i in sent_tokenize(f): 
    temp = [] 
      
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    data.append(temp) 
  
# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5) 
  
# Print results 
print("dog cat : ", model1.similarity('dog', 'cat')) 
print("cat dog : ", model1.similarity('cat', 'dog')) 
print("dog dog : ", model1.similarity('dog', 'dog')) 
print("dog eat : ", model1.similarity('dog', 'eat')) 
print("chase eat : ", model1.similarity('chase', 'eat'))
print("chase a : ", model1.similarity('chase', 'a')) 
print("dog a : ", model1.similarity('dog', 'a')) 
