from transformers import pipeline

# Allocate a pipeline for question-answering
question_answerer = pipeline('question-answering')
result = question_answerer({
    'question': 'What is the name of the repository ?',
    'context': 'Pipeline has been included in the huggingface/transformers repository'
})
print(result) # {'score': 0.30970096588134766, 'start': 34, 'end': 58, 'answer': 'huggingface/transformers'}
