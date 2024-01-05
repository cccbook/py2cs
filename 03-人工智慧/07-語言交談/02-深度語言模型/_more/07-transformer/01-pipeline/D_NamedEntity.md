# pipeline D

Here is an example of doing named entity recognition, using a model and a tokenizer. 

```
$ python pipelineD.py
Downloading: 100%|███████████████████████████████████████████████████| 29.0/29.0 [00:00<00:00, 9.62kB/s]
Downloading: 100%|█████████████████████████████████████████████████████| 570/570 [00:00<00:00, 61.9kB/s]
Downloading: 100%|████████████████████████████████████████████████████| 208k/208k [00:01<00:00, 157kB/s]
Downloading: 100%|████████████████████████████████████████████████████| 426k/426k [00:02<00:00, 200kB/s]
('[CLS]', 'O')
('Hu', 'I-ORG')
('##gging', 'I-ORG')
('Face', 'I-ORG')
('Inc', 'I-ORG')
('.', 'O')
('is', 'O')
('a', 'O')
('company', 'O')
('based', 'O')
('in', 'O')
('New', 'I-LOC')
('York', 'I-LOC')
('City', 'I-LOC')
('.', 'O')
('Its', 'O')
('headquarters', 'O')
('are', 'O')
('in', 'O')
('D', 'I-LOC')
('##UM', 'I-LOC')
('##BO', 'I-LOC')
(',', 'O')
('therefore', 'O')
('very', 'O')
('close', 'O')
('to', 'O')
('the', 'O')
('Manhattan', 'I-LOC')
('Bridge', 'I-LOC')
('.', 'O')
('[SEP]', 'O')
```