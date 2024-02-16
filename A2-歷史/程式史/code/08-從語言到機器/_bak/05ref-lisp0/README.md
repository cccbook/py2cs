# lis.py

來源 -- http://norvig.com/lispy.html

```
ccckmit@asus MINGW64 /d/ccc/ccc112a/py2cs/06-計算理論/code/08-從語言到機器/05-lisp0 (master)
$ python lis.py
tokenize: ['(', 'begin', '(', 'define', 'r', '10', ')', '(', '*', 'pi', '(', '*', 'r', 'r', ')', ')', ')']
parse: ['begin', ['define', 'r', 10], ['*', 'pi', ['*', 'r', 'r']]]
eval 314.1592653589793

ccckmit@asus MINGW64 /d/ccc/ccc112a/py2cs/06-計算理論/code/08-從語言到機器/05-lisp0 (master)
$ python listest.py
(quote (testing 1 (2.0) -3.14e159)) => (testing 1 (2.0) -3.14e+159)
(+ 2 2) => 4
(+ (* 2 100) (* 1 10)) => 210
(if (> 6 5) (+ 1 1) (+ 2 2)) => 2
(if (< 6 5) (+ 1 1) (+ 2 2)) => 4
(define x 3) => None
x => 3
(+ x x) => 6
(begin (define x 1) (set! x (+ x 1)) (+ x 1)) => 3
((lambda (x) (+ x x)) 5) => 10
(define twice (lambda (x) (* 2 x))) => None
(twice 5) => 10
(define compose (lambda (f g) (lambda (x) (f (g x))))) => None
((compose list twice) 5) => (10)
(define repeat (lambda (f) (compose f f))) => None
((repeat twice) 5) => 20
((repeat (repeat twice)) 5) => 80
(define fact (lambda (n) (if (<= n 1) 1 (* n (fact (- n 1)))))) => None
(fact 3) => 6
(fact 50) => 30414093201713378043612608166064768844377641568960512000000000000
(define abs (lambda (n) ((if (> n 0) + -) 0 n))) => None
(list (abs -3) (abs 0) (abs 3)) => (3 0 3)
(define combine (lambda (f)
    (lambda (x y)
      (if (null? x) (quote ())
          (f (list (car x) (car y))
             ((combine f) (cdr x) (cdr y))))))) => None
(define zip (combine cons)) => None
(zip (list 1 2 3 4) (list 5 6 7 8)) => ((1 5) (2 6) (3 7) (4 8))
(define riff-shuffle (lambda (deck) (begin
    (define take (lambda (n seq) (if (<= n 0) (quote ()) (cons (car seq) (take (- n 1) (cdr seq))))))
    (define drop (lambda (n seq) (if (<= n 0) seq (drop (- n 1) (cdr seq)))))
    (define mid (lambda (seq) (/ (length seq) 2)))
    ((combine append) (take (mid deck) deck) (drop (mid deck) deck))))) => None
(riff-shuffle (list 1 2 3 4 5 6 7 8)) => (1 5 2 6 3 7 4 8)
((repeat riff-shuffle) (list 1 2 3 4 5 6 7 8)) => (1 3 5 7 2 4 6 8)
(riff-shuffle (riff-shuffle (riff-shuffle (list 1 2 3 4 5 6 7 8)))) => (1 2 3 4 5 6 7 8)
********************************************* lis.py: 0 out of 29 tests fail.
```