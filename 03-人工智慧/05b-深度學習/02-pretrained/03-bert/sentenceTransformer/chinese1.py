from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

wordpairs = [['老師', '教師', '泰國'], 
             ['商品', '貨物', '跑步']]

for wordpair in wordpairs:
    embeddings = model.encode(wordpair)
    print(wordpair[0], 'vs',  wordpair[1], 'distance =', util.pytorch_cos_sim(embeddings[0], embeddings[1]))
    print(wordpair[0], 'vs',  wordpair[2], 'distance =', util.pytorch_cos_sim(embeddings[0], embeddings[2]))
    print(wordpair[1], 'vs',  wordpair[2], 'distance =', util.pytorch_cos_sim(embeddings[1], embeddings[2]))
