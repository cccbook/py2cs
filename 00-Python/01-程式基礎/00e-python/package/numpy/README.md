# numpy 數值套件

```
PS D:\ccc\course\ai\python\02-optimize> python
Python 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 19:29:22) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> a = np.arange(0,4)
>>> b = np.arange(1,5)
>>> np.add(a,b)
array([1, 3, 5, 7])
>>> a+b
array([1, 3, 5, 7])
>>> a>b
array([False, False, False, False])
>>> a*b
array([ 0,  2,  6, 12])
>>> a
array([0, 1, 2, 3])
>>> b
array([1, 2, 3, 4])
>>> a.shape
(4,)
>>> a.shape = (2,2)
>>> a
array([[0, 1],
       [2, 3]])
>>> a.reshape((4,))
array([0, 1, 2, 3])
>>> a
array([[0, 1],
       [2, 3]])
>>> a.shape = (4,)
>>> a
array([0, 1, 2, 3])
>>> a.dtype
dtype('int32')
>>> a[4] = 5
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index 4 is out of bounds for axis 0 with size 4
>>>
>>> e = np.linspace(0, 1, 10)
>>> e
array([0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
       0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ])
>>> f = np.logspace(0, 2, 5)
>>> f
array([  1.        ,   3.16227766,  10.        ,  31.6227766 ,
       100.        ])
>>> g = np.zeros(4)
>>> g
array([0., 0., 0., 0.])
>>> g.dtype
dtype('float64')
>>> a
array([0, 1, 2, 3])
>>> a[2]
2
>>> a[2:4]
array([2, 3])
>>> a[1:4:2]
array([1, 3])
>>> a[::-1]
array([3, 2, 1, 0])
>>> x = np.random.randint(0, 10, 6)
>>> x
array([5, 4, 1, 9, 4, 5])
>>> x[x>5]
array([9])
>>> x[x>4]
array([5, 9, 5])
```
