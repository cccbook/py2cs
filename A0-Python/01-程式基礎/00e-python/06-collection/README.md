# python 集合物件

## list

就像陣列

```py
shoplist = ['apple', 'mango', 'carrot', 'banana']

list[from:to:step] 其中的 step 代表每次跳幾格：

>>> shoplist = ['apple', 'mango', 'carrot', banana']
>>> shoplist[::1]
['apple', 'mango', 'carrot', 'banana']
>>> shoplist[::2]
['apple', 'carrot']
>>> shoplist[::3]
['apple', 'banana']
>>> shoplist[::-1]
['banana', 'carrot', 'mango', 'apple']
```

## tuple

不可修改的陣列

```
zoo = ('python', 'elephant', 'penguin')
```

## Dictionary

字典

```py
ab = :
    'Swaroop': 'swaroop@swaroopch.com',
    'Larry': 'larry@wall.org',
    'Matsumoto': 'matz@ruby-lang.org',
    'Spammer': 'spammer@hotmail.com'

```

## Set

```py
>>> bri = set(['brazil', 'russia', 'india'])
>>> 'india' in bri
True
>>> 'usa' in bri
False
>>> bric = bri.copy()
>>> bric.add('china')
>>> bric.issuperset(bri)
True
>>> bri.remove('russia')
>>> bri & bric # OR bri.intersection(bric)
:'brazil', 'india'
```

## Sequence

可以做 membership test.

Lists, tuples and strings are examples of sequences

