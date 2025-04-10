from transformers import AutoTokenizer, AutoModelForCausalLM

access_token = "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"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", token=access_token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", token=access_token)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))