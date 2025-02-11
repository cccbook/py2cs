from transformers import pipeline

generator = pipeline("text-generation") # model 預設為 openai-community/gpt2
# r = generator("In this course, we will teach you how to")
r = generator("What is a doctor? A doctor is a ")

print(r)
