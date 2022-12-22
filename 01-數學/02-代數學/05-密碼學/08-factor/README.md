# 因數分解


## 試除法

* [維基百科: 試除法](https://zh.wikipedia.org/wiki/%E8%AF%95%E9%99%A4%E6%B3%95)

```
PS D:\ccc\ccc109a\se\deno\alg\14-cryptography\factor> deno run factor.js
factor(15)= 3
factor(37)= -1
factor(9373467139)= 2141
factor(10000819)= -1    
factor(3093215881333057)= -1
factor(489133282872437279)= 2
489133282872437279= 489133282872437250       
489133282872437279.000001= 489133282872437250
```

## 米勒-拉賓質數判定法

* [維基百科: 米勒-拉賓質數判定法](https://zh.wikipedia.org/wiki/%E7%B1%B3%E5%8B%92-%E6%8B%89%E5%AE%BE%E6%A3%80%E9%AA%8C)

```
PS D:\ccc109\se\python\alg\14-cryptography_todo\factor> python millerRabin.py
probablyPrime(37)= True
probablyPrime(33)= False
```

## AKS

* https://en.wikipedia.org/wiki/AKS_primality_test 
    * 理論上多項式時間質數測試，不依賴黎曼猜想

```
PS D:\ccc109\se\python\alg\14-cryptography_todo\factor> python aks.py
# p: (x-1)^p for small p
  0: +1
  1: -1 +1x^1
  2: +1 -2x^1 +1x^2
  3: -1 +3x^1 -3x^2 +1x^3
  4: +1 -4x^1 +6x^2 -4x^3 +1x^4
  5: -1 +5x^1 -10x^2 +10x^3 -5x^4 +1x^5
  6: +1 -6x^1 +15x^2 -20x^3 +15x^4 -6x^5 +1x^6
  7: -1 +7x^1 -21x^2 +35x^3 -35x^4 +21x^5 -7x^6 +1x^7
  8: +1 -8x^1 +28x^2 -56x^3 +70x^4 -56x^5 +28x^6 -8x^7 +1x^8
  9: -1 +9x^1 -36x^2 +84x^3 -126x^4 +126x^5 -84x^6 +36x^7 -9x^8 +1x^9
 10: +1 -10x^1 +45x^2 -120x^3 +210x^4 -252x^5 +210x^6 -120x^7 +45x^8 -10x^9 +1x^10
 11: -1 +11x^1 -55x^2 +165x^3 -330x^4 +462x^5 -462x^6 +330x^7 -165x^8 +55x^9 -11x^10 +1x^11

# small primes using the aks test
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
```


* http://rosettacode.org/wiki/AKS_test_for_primes#Python

The AKS algorithm for testing whether a number is prime is a polynomial-time algorithm based on an elementary theorem about Pascal triangles.

The theorem on which the test is based can be stated as follows:

* a number  p  is prime   if and only if   all the coefficients of the polynomial expansion of*

```math
(x-1)^{p}-(x^{p}-1)
```

are divisible by.

* https://stackoverflow.com/questions/347811/aks-primes-algorithm-in-python

```
Quick answer: no, the AKS test is not the fastest way to test primality. There are much much faster primality tests that either assume the (generalized) Riemann hypothesis and/or are randomized. (E.g. Miller-Rabin is fast and simple to implement.) The real breakthrough of the paper was theoretical, proving that a deterministic polynomial-time algorithm exists for testing primality, without assuming the GRH or other unproved conjectures.

That said, if you want to understand and implement it, Scott Aaronson's short article might help. It doesn't go into all the details, but you can start at page 10 of 12, and it gives enough. :-) There is also a list of implementations (mostly in C++) here.

Also, for optimization and improvements (by several orders of magnitude), you might want to look at this report, or (older) Crandall and Papadopoulos's report, or (older still) Daniel J Bernstein's report. All of them have fairly detailed pseudo-code that lends itself well to implementation.
```
