# C(n,k)

## Cnk.py

```
$ python Cnk.py
c(5,2)= 10.0
c(7,3)= 35.0
c(12,5)= 792.0
c(60,30)= 1.1826458156486142e+17
```

## CnkR.py

```
$ python CnkR.py
c(5,2)= 10
c(7,3)= 35  
c(12,5)= 792
Traceback (most recent call last):
  File "CnkR.py", line 10, in <module>
    print("c(60,30)=", c(60,30))
  File "CnkR.py", line 5, in c
    return c(n-1, k) + c(n-1, k-1)
  File "CnkR.py", line 5, in c
    return c(n-1, k) + c(n-1, k-1)
  File "CnkR.py", line 5, in c
    return c(n-1, k) + c(n-1, k-1)
  [Previous line repeated 56 more times]
KeyboardInterrupt
```

## CnkRLookup.py

```
$ python CnkRLookup.py
c(5,2)= 10
c(7,3)= 35
c(12,5)= 792
c(60,30)= 118264581564861424
```
