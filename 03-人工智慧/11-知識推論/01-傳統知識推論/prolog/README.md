# Prolog

* https://www.ruanyifeng.com/blog/2019/01/prolog.html
* https://zh.wikipedia.org/wiki/Prolog
* https://www.cpp.edu/~jrfisher/www/prolog_tutorial/2.html (讚)
* https://www.cs.ccu.edu.tw/~dan/prologProgs.html

## friend.pl

```

?- [friend].
true.

?- friend(john, jack).
true.

?- friend(john, sam).
false.

?- listing(friend).
friend(john, julia).
friend(john, jack).
friend(julia, sam).
friend(julia, molly).

true.

?- friend(john, Who).
Who = julia ;
Who = jack.
```

map.pl

```
?- [map].
true.

?- colorify(A,B,C,D,E).
A = red,
B = D, D = green,
C = E, E = blue ;
A = red,
B = D, D = blue,
C = E, E = green ;
A = green,
B = D, D = red,
C = E, E = blue ;
A = green,
B = D, D = blue,
C = E, E = red ;
A = blue,
B = D, D = red,
C = E, E = green ;
A = blue,
B = D, D = green,
C = E, E = red ;
false.
```

nqueens.pl

```
?- [nqueens].
true.

?- queens(4, Qs).
Qs = [[2, 4, 1, 3], [3, 1, 4, 2]].

?- queens(8, Qs).
Qs = [[1, 5, 8, 6, 3, 7, 2, 4], [1, 6, 8, 3, 7, 4, 2|...], [1, 7, 4, 6, 8, 2|...], [1, 7, 5, 8, 2|...], [2, 4, 6, 8|...], [2, 5, 7|...], [2, 5|...], [2|...], [...|...]|...].

```

fact.pl

```

?- [fact].
true.

?- fact(5,R).
R = 120.
```

family.pl

* https://cse.sc.edu/~ahein/330/example.html

```
?- [family].
true.

?- parent(john,X).
false.

?- parent(X,john).
X = sue ;
X = sam.

?- grand_father(X,john).
X = george ;
false.
```

hanoi.pl

* https://www.cpp.edu/~jrfisher/www/prolog_tutorial/2_3.html

```
?- [hanoi].
true.

?- move(3,left,right,center). 
Move top disk from left to right
Move top disk from left to center
Move top disk from right to center
Move top disk from left to right
Move top disk from center to left
Move top disk from center to right
Move top disk from left to right
true 
```