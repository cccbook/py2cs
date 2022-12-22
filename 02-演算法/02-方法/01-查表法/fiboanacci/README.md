# 用查表加速 -- 以費氏數列為例

傳統用遞迴方式的費氏數列算法，會耗費很久的時間：

```
$ python fibonacci.py 
fibonacci(35)=9227465
time:0:00:16.081451
```

加入查表，讓已經算過的就不需要算第二次，第二次之後改用查的，就會變很快！

```
$ python fibonacci_lookup.py 
fibonacci(35)=9227465
time:$0:00:00.002593
```
