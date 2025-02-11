# 統計

```py
PS D:\ccc\course\ai\python\08-scientific\11-statistics> python
Python 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> from statistics import *
>>> mean([1, 2, 3, 4, 4])
2.8
>>> mean([-1.0, 2.5, 3.25, 5.75])
2.625
>>> median([1, 3, 5])
3
>>> median([1, 3, 5, 7])
4.0
>>> mode([1, 1, 2, 3, 3, 3, 3, 4])
3
>>> mode(["red", "blue", "blue", "red", "green", "red", "red"])
'red'
>>> data = [0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25]
>>> pvariance(data)
1.25
>>> variance(data)
1.4285714285714286
>>> stdev(data)
1.1952286093343936
>>> pstdev(data)
1.118033988749895
```


## 參考

* [數據分析必掌握的統計學知識！](https://medium.com/%E6%95%B8%E6%93%9A%E5%88%86%E6%9E%90%E4%B8%8D%E6%98%AF%E5%80%8B%E4%BA%8B/%E6%95%B8%E6%93%9A%E5%88%86%E6%9E%90%E5%BF%85%E6%8E%8C%E6%8F%A1%E7%9A%84%E7%B5%B1%E8%A8%88%E5%AD%B8%E7%9F%A5%E8%AD%98-6edf53d98407)