# pipeline C

named entity recognition

```
$ python pipelineC.py
2022-03-07 10:19:13.292295: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-07 10:19:13.292804: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english (https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english)
Downloading: 100%|██████████████████████████████████████████████████████| 998/998 [00:00<00:00, 332kB/s] 
Downloading: 100%|█████████████████████████████████████████████████| 1.24G/1.24G [09:10<00:00, 2.43MB/s] 
Downloading: 100%|███████████████████████████████████████████████████| 60.0/60.0 [00:00<00:00, 5.97kB/s]
Downloading: 100%|████████████████████████████████████████████████████| 208k/208k [00:01<00:00, 157kB/s]
{'entity': 'I-ORG', 'score': 0.99957865, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': 0.9909764, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7} 
{'entity': 'I-ORG', 'score': 0.9982224, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}   
{'entity': 'I-ORG', 'score': 0.9994879, 'index': 4, 'word': 'Inc', 'start': 13, 'end': 16}   
{'entity': 'I-LOC', 'score': 0.9994344, 'index': 11, 'word': 'New', 'start': 40, 'end': 43}  
{'entity': 'I-LOC', 'score': 0.99931955, 'index': 12, 'word': 'York', 'start': 44, 'end': 48}
{'entity': 'I-LOC', 'score': 0.9993794, 'index': 13, 'word': 'City', 'start': 49, 'end': 53}
{'entity': 'I-LOC', 'score': 0.98625815, 'index': 19, 'word': 'D', 'start': 79, 'end': 80}
{'entity': 'I-LOC', 'score': 0.95142686, 'index': 20, 'word': '##UM', 'start': 80, 'end': 82}
{'entity': 'I-LOC', 'score': 0.9336589, 'index': 21, 'word': '##BO', 'start': 82, 'end': 84}
{'entity': 'I-LOC', 'score': 0.9761654, 'index': 28, 'word': 'Manhattan', 'start': 114, 'end': 123}      
{'entity': 'I-LOC', 'score': 0.9914629, 'index': 29, 'word': 'Bridge', 'start': 124, 'end': 130}
```
