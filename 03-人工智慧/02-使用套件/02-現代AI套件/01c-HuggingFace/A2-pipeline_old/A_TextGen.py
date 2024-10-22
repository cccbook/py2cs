from transformers import pipeline

text_generator = pipeline("text-generation")
print(text_generator("As far as I am concerned, I will", max_length=50, do_sample=False))
# [{'generated_text': 'As far as I am concerned, I will be the first to admit that I am not a fan of the idea of a
# "free market." I think that the idea of a free market is a bit of a stretch. I think that the idea'}]
