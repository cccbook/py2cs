from transformers import BertTokenizer #  BertModel, BertConfig, 
'''
# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig()

# Initializing a model from the bert-base-uncased style configuration
model = BertModel(configuration)

# Accessing the model configuration
configuration = model.config
'''
tokenizer = BertTokenizer.from_pretrained("hfl/BERT-wwm")
model = BertModel.from_pretrained("hfl/BERT-wwm")