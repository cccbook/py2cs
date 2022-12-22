# article-permutation-circuit-synthesis
This article describes how embedded languages and recursion can be used to create a tool that synthesizes a relatively efficient logical circuit for any chosen permutation of the set of all bit vectors of some fixed length.

## naive.py

```
$ python naive.py
100%|████████████████| 256/256 [00:13<00:00, 19.20it/s]
True
{(0, 0, 0, 1): 7168, (0, 1, 1, 1): 1016, (1, 0): 3711}  
```


## naive.py

```
$ python optimized.py
100%|███████████████| 256/256 [00:00<00:00, 293.33it/s]
True
{(0, 0, 0, 1): 303, (0, 1, 1, 1): 494, (1, 0): 16}   
```
