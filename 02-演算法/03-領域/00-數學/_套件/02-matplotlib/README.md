# Matplotlib 繪圖套件


## mgrid 函數

https://docs.scipy.org/doc/numpy/reference/generated/numpy.mgrid.html

if the step length is a complex number (e.g. 5j), then the integer part of its magnitude is interpreted as specifying the number of points to create between the start and stop values,

這個語法很奇怪， mgrid 看起來像個陣列 (不是應該是函數嗎？ 該怎麼寫出可以這樣呼叫的函數呢？這應該是我不懂的技巧)

```py
>>> np.mgrid[0:5,0:5]
array([[[0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4]],
       [[0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]]])
>>> np.mgrid[-1:1:5j]
array([-1. , -0.5,  0. ,  0.5,  1. ])
```

## 3d 繪圖

範例 curve3d 修改加入 `from mpl_toolkits.mplot3d import Axes3D` 之後就成功了！

https://stackoverflow.com/questions/3810865/matplotlib-unknown-projection-3d-error

@dashesy - You'll still need to import the 3d projections: from mpl_toolkits.mplot3d import Axes3D. Afterwards, it should work. – Joe Kington Apr 24 '13 at 2:46