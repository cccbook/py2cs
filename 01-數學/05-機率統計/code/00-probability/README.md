# Probability



## numpy.random

* https://docs.scipy.org/doc/numpy-1.14.1/reference/routines.random.html

```
PS D:\ccc\course\ai\python\08-scientific\11-statistics> python
Python 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> mu, sigma = 5, 2
>>> np.random.normal(mu, sigma, 10)
array([5.46227641, 3.57734384, 6.63598313, 5.67136977, 5.56860855,
       1.62940233, 0.27267601, 7.11307582, 7.59549649, 3.00867069])
>>> s = np.random.normal(mu, sigma, 10)
>>> np.mean(s)
5.402271097544764
>>> np.std(s)
2.3714910869544825
>>> su = np.random.uniform(2,5,10)
>>> min, max=2,5
>>> su = np.random.uniform(min,max,10)
>>> su
array([3.05949709, 4.40338278, 2.41100464, 3.98757621, 2.76181156,
       3.80287061, 2.59318563, 4.61296937, 3.37523727, 4.58688005])
>>> np.mean(s)
5.402271097544764
>>> np.mean(su)
3.559441520076601
>>> rand(3,5)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'rand' is not defined
>>> np.random.rand(3,5)
array([[0.34763696, 0.62619926, 0.99862022, 0.28091175, 0.19166789],
       [0.07549228, 0.78721126, 0.02703787, 0.56109626, 0.17058729],
       [0.09977371, 0.61782206, 0.03610409, 0.84066091, 0.41094005]])
>>> np.random.rand(3)
array([0.9362097 , 0.99323004, 0.20741079])
>>> np.random.rand(2,4)
array([[0.98993393, 0.55385856, 0.4878733 , 0.76579933],
       [0.0597472 , 0.38233857, 0.94057392, 0.5208569 ]])
>>> np.random.rand(2,2,2)
array([[[0.72707447, 0.84577164],
        [0.21603509, 0.93999889]],

       [[0.12673862, 0.46966462],
        [0.25480117, 0.07714917]]])
>>> np.random.randn(2,4)
array([[ 0.20208997, -1.71604126,  0.12405509, -0.24746528],
       [-0.12728393,  1.79975707,  1.3082333 , -0.20118368]])
>>> R = np.random
>>> R.randn(2,4)
array([[ 0.79089131, -0.12458133, -0.23805414,  0.36160677],
       [ 0.69552804, -0.62223268,  1.46829889, -1.86481991]])
>>> R.choice([1,2,3,4,5,6], 5)
array([5, 6, 6, 3, 3])
>>> R.choice([1,2,3,4,5,6], 10)
array([1, 2, 6, 1, 5, 6, 5, 1, 4, 5])
>>> R.bytes(10)
b'pU\x12e\x99\xcc,\xac\xe7\xbf'
>>> R.choice([0,1], 10)
array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
>>> R.choice([0,1], 10)
array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0])
```

## scipy.stats

## matplotlib.pyplot

* https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html

## Shape

```
PS D:\ccc\course\ai\python\08-scientific\12-probability> python
Python 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> s = np.array([1,2,3,4])
>>> s.shape
(4,)
>>> s.reshape((2,2))
array([[1, 2],
       [3, 4]])
>>> s = np.array([1.1,2.2,3.3,4.4])
>>> s.reshape((2,2))
array([[1.1, 2.2],
       [3.3, 4.4]])
>>> s = np.array([1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8])
>>> s.reshape((2,4))
array([[1.1, 2.2, 3.3, 4.4],
       [5.5, 6.6, 7.7, 8.8]])
>>> s
array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8])
>>> s.reshape((2,4)).sum(axis=0)
array([ 6.6,  8.8, 11. , 13.2])
>>> s.reshape((2,4))
array([[1.1, 2.2, 3.3, 4.4],
       [5.5, 6.6, 7.7, 8.8]])
>>> s.reshape((2,4)).sum(axis=0)
array([ 6.6,  8.8, 11. , 13.2])
>>> exit()
```