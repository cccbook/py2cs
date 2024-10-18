from transformers import pipeline

text = "translate English to Chinese: Hugging Face is a community-based open-source platform for machine learning."
translator = pipeline(task="translation", model="t5-small")

print(text)
print(translator(text))
