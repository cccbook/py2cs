# rnn_qa

## qa

```
$ python main.py qa qa
tokens= 8362  
len(ids)= 8362
ids.size(0)= 8362
batch_size= 20
num_batches= 418
len(ids)= 8360
ids.shape= torch.Size([20, 418])
vocab_size= 20
['Q:', 'the', 'cat', 'love', 'every', 'dog', 'A:'] ['5']
['Q:', 'the', 'little', 'pig', 'A:'] ['6']
['Q:', 'every', 'little', 'dog', 'A:'] ['7']
['Q:', 'the', 'little', 'pig', 'bite', 'a', 'black', 'cat', 'A:'] ['7']
['Q:', 'a', 'white', 'cat', 'love', 'the', 'black', 'dog', 'A:'] ['7']
['Q:', 'every', 'pig', 'bite', 'every', 'white', 'cat', 'A:'] ['7']
['Q:', 'the', 'little', 'dog', 'bite', 'the', 'little', 'cat', 'A:'] ['7']
['Q:', 'the', 'white', 'pig', 'bite', 'a', 'white', 'pig', 'A:'] ['7']
```

## summary

```
$ python main.py summary qa
tokens= 9241
len(ids)= 9241
ids.size(0)= 9241
batch_size= 20
num_batches= 462
len(ids)= 9240
ids.shape= torch.Size([20, 462])
vocab_size= 15
['Q:', 'the', 'cat', 'love', 'every', 'dog', 'A:'] ['cat', 'love', 'dog']
['Q:', 'the', 'little', 'pig', 'A:'] ['dog', 'bite', 'pig']
['Q:', 'every', 'little', 'dog', 'A:'] ['dog']
['Q:', 'the', 'little', 'pig', 'bite', 'a', 'black', 'cat', 'A:'] ['pig', 'love', 'cat']       
['Q:', 'a', 'white', 'cat', 'love', 'the', 'black', 'dog', 'A:'] ['cat', 'love', 'dog']        
['Q:', 'every', 'pig', 'bite', 'every', 'white', 'cat', 'A:'] ['pig', 'chase', 'cat']
['Q:', 'the', 'little', 'dog', 'bite', 'the', 'little', 'cat', 'A:'] ['dog', 'chase', 'cat']   
['Q:', 'the', 'white', 'pig', 'bite', 'a', 'white', 'pig', 'A:'] ['pig', 'bite', 'pig']   
```