# TorchText

* https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
* https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html

## 01-Classify

```
(env) mac020:01-classify1 mac020$ python torchtext_classify.py
120000lines [00:13, 9127.25lines/s]
120000lines [00:25, 4773.44lines/s]
7600lines [00:01, 4880.03lines/s]
Epoch: 1  | time in 0 minutes, 33 seconds
        Loss: 0.0261(train)     |       Acc: 84.8%(train)
        Loss: 0.0001(valid)     |       Acc: 90.6%(valid)
Checking the results of test dataset...
Epoch: 2  | time in 0 minutes, 32 seconds
        Loss: 0.0117(train)     |       Acc: 93.7%(train)
        Loss: 0.0002(valid)     |       Acc: 90.3%(valid)
Checking the results of test dataset...
Epoch: 3  | time in 0 minutes, 32 seconds
        Loss: 0.0067(train)     |       Acc: 96.5%(train)
        Loss: 0.0002(valid)     |       Acc: 90.9%(valid)
Checking the results of test dataset...
Epoch: 4  | time in 0 minutes, 33 seconds
        Loss: 0.0037(train)     |       Acc: 98.2%(train)
        Loss: 0.0001(valid)     |       Acc: 90.4%(valid)
Checking the results of test dataset...
Epoch: 5  | time in 0 minutes, 33 seconds
        Loss: 0.0022(train)     |       Acc: 99.0%(train)
        Loss: 0.0002(valid)     |       Acc: 90.7%(valid)
Checking the results of test dataset...
        Loss: 0.0002(test)      |       Acc: 89.4%(test)
This is a Sports news
```

