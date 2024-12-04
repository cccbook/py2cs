from transformers import pipeline

generator = pipeline("text-generation", model="openai-community/gpt2", device="mps") # model 預設為 openai-community/gpt2
r = generator("In this course, we will teach you how to")
print(r)