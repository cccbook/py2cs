# Lambda Calculus 當中的所有資料結構都是用 closure 達成的
# 換言之，所有的資料結構都是函數閉包。

# Church Booleans : Logic
IF    = lambda c:lambda x:lambda y:c(x)(y) #  if: λ c x y. c x y # if c then x else y.
TRUE  = lambda x:lambda y:x # if true then x # 兩個參數執行第一個
FALSE = lambda x:lambda y:y # if false then y # 兩個參數執行第二個
AND   = lambda p:lambda q:p(q)(p) # if p then q else p
OR    = lambda p:lambda q:p(p)(q) # if p then p else q
XOR   = lambda p:lambda q:p(NOT(q))(q) #  if p then not q else q
NOT   = lambda c:c(FALSE)(TRUE) # if c then false else true

ASSERT = lambda truth: (IF(truth)
    (lambda description:f'[✓] ${description}')
    (lambda description:f'[✗] ${description}')
)

REFUTE = lambda truth:ASSERT(NOT(truth))

TEST   = lambda description:lambda assertion:\
    print(assertion(description))

TEST('TRUE')\
    (ASSERT(TRUE))

TEST('FALSE')\
    (REFUTE(FALSE))

TEST('AND')\
  (ASSERT(AND(TRUE)(TRUE)))

TEST('OR')(ASSERT(AND\
  (AND(OR(TRUE)(FALSE))(OR(FALSE)(TRUE)))\
  (NOT(OR(FALSE)(FALSE)))))

TEST('XOR')(ASSERT(AND\
  (AND(XOR(TRUE)(FALSE))(XOR(FALSE)(TRUE)))\
  (NOT(XOR(TRUE)(TRUE)))))

TEST('NOT')\
  (REFUTE(NOT(TRUE)))

# Arithmetics
IDENTITY       = lambda x:x
SUCCESSOR      = lambda n:lambda f:lambda x:f(n(f)(x))
PREDECESSOR    = lambda n:lambda f:lambda x:n(lambda g : lambda h : h(g(f)))(lambda _ : x)(lambda u : u)
ADDITION       = lambda m:lambda n:n(SUCCESSOR)(m)
SUBTRACTION    = lambda m:lambda n:n(PREDECESSOR)(m)
MULTIPLICATION = lambda m:lambda n:lambda f:m(n(f))
POWER          = lambda x:lambda y:y(x)
ABS_DIFFERENCE = lambda x:lambda y:ADDITION(SUBTRACTION(x)(y))(SUBTRACTION(y)(x))

# Church Numerals
_zero  = lambda f:IDENTITY # 0      : 用 λf. λx. x 當 0
_one   = SUCCESSOR(_zero)  # 1=S(0) : λf. λf. λx. x 當 1
_two   = SUCCESSOR(_one)   # 2=S(1) : λf. λf. λf. λx. x 當 2
_three = SUCCESSOR(_two)   # 3=S(2)
_four  = MULTIPLICATION(_two)(_two)  # 4 = 2*2
_five  = SUCCESSOR(_four)            # 5 = S(4)
_eight = MULTIPLICATION(_two)(_four) # 8 = 2*4
_nine  = SUCCESSOR(_eight)           # 9 = S(8)
_ten   = MULTIPLICATION(_two)(_five) # 10 = 2*5

# Comparison
IS_ZERO               = lambda n:n(lambda _:FALSE)(TRUE)
IS_LESS_THAN          = lambda m:lambda n:NOT(IS_LESS_THAN_EQUAL(n)(m))
IS_LESS_THAN_EQUAL    = lambda m:lambda n:IS_ZERO(SUBTRACTION(m)(n))
IS_EQUAL              = lambda m:lambda n:AND(IS_LESS_THAN_EQUAL(m)(n))(IS_LESS_THAN_EQUAL(n)(m))
IS_NOT_EQUAL          = lambda m:lambda n:OR(NOT(IS_LESS_THAN_EQUAL(m)(n)))(NOT(IS_LESS_THAN_EQUAL(n)(m)))
IS_GREATER_THAN_EQUAL = lambda m:lambda n:IS_LESS_THAN_EQUAL(n)(m)
IS_GREATER_THAN       = lambda m:lambda n:NOT(IS_LESS_THAN_EQUAL(m)(n))
IS_NULL               = lambda p:p(lambda x:lambda y:FALSE)
NIL                   = lambda x:TRUE

# 另一種定義方式 Y = lambda g:g(lambda:Y(g))
# Y-Combinators : 令 g = (lambda x:f(lambda y:x(x)(y)))
# 利用遞迴的方式 Y(g) = g(g) = g(g)(y) = 
Y = lambda f:\
  (lambda x:f(lambda y:x(x)(y)))\
  (lambda x:f(lambda y:x(x)(y)))

# Lists

CONS = lambda x:lambda y:lambda f:f(x)(y) # 將 x,y 形成配對 PAIR(x,y)
CAR  = lambda p:p(TRUE) # 取得 PAIR(x,y) 中的頭部 x (HEAD)
CDR  = lambda p:p(FALSE) # 取得 PAIR(x,y) 中的尾部 y (TAIL)

# ==== 以下為常用的 functional programming 函數 ====
RANGE = lambda m:lambda n:Y(lambda f:lambda m:IF(IS_EQUAL(m)(n))\
  (lambda _: CONS(m)(NIL))\
  (lambda _: CONS(m)(f(SUCCESSOR(m))))\
(NIL))(m)
# 令 g = lambda f:lambda m:IF(IS_EQUAL(m)(n))\
#          (lambda _: CONS(m)(NIL))\
#          (lambda _: CONS(m)(f(SUCCESSOR(m))))\
#          (NIL)
# 再令 h = (lambda x:f(lambda y:x(x)(y)))
# Y(g) = h(h) = f(lambda y:h(h)(y))
# RANGE(3)(5) = Y(g)(3)(5) = g(g)(Y)(3)(5) = CONS(3)(g(m+1)) ...

# print(RANGE(_three))
# print(RANGE(_three)(_five))

MAP = lambda x:lambda g:Y(lambda f:lambda x:IF(IS_NULL(x))\
  (lambda _: x)\
  (lambda _: CONS(g(CAR(x)))(f(CDR(x))))\
(NIL))(x)

TEST('IDENTITY')\
  (ASSERT(IS_EQUAL(IDENTITY)(lambda x:x)))

TEST('SUCCESSOR')\
  (ASSERT(IS_EQUAL(SUCCESSOR(_zero))(_one)))

TEST('PREDECESSOR')\
  (ASSERT(IS_EQUAL(_zero)(PREDECESSOR(_one))))

TEST('ADDITION')\
  (ASSERT(IS_EQUAL(SUCCESSOR(_one))(ADDITION(_one)(_one))))

TEST('SUBTRACTION')\
  (ASSERT(IS_EQUAL(_zero)(SUBTRACTION(_one)(_one))))

TEST('MULTIPLICATION')\
  (ASSERT(IS_EQUAL(_four)(MULTIPLICATION(_two)(_two))))

TEST('POWER')(ASSERT(AND\
  (IS_EQUAL(_nine)(POWER(_three)(_two)))\
  (IS_EQUAL(_eight)(POWER(_two)(_three)))))

TEST('ABS_DIFFERENCE')(ASSERT(AND\
  (IS_EQUAL(_one)(ABS_DIFFERENCE(_three)(_two)))\
  (IS_EQUAL(_one)(ABS_DIFFERENCE(_two)(_three)))))

TEST('IS_ZERO')\
  (ASSERT(IS_ZERO(_zero)))

TEST('IS_LESS_THAN')\
  (ASSERT(IS_LESS_THAN(_zero)(_one)))

TEST('IS_LESS_THAN_EQUAL')(ASSERT(AND\
  (IS_LESS_THAN_EQUAL(_one)(_one))\
  (IS_LESS_THAN_EQUAL(_zero)(_one))))

TEST('IS_EQUAL')(ASSERT(AND\
  (IS_EQUAL(_zero)(_zero))\
  (IS_EQUAL(_one)(_one))))

TEST('IS_NOT_EQUAL')\
  (ASSERT(IS_NOT_EQUAL(_zero)(_one)))

TEST('IS_GREATER_THAN_EQUAL')(ASSERT(AND\
  (IS_GREATER_THAN_EQUAL(_one)(_one))\
  (IS_GREATER_THAN_EQUAL(_one)(_zero))))

TEST('IS_GREATER_THAN')\
  (ASSERT(IS_GREATER_THAN(_one)(_zero)))

TEST('IS_NULL')\
  (ASSERT(IS_NULL(NIL)))

TEST('CAR')(ASSERT(AND\
  (IS_EQUAL(CAR(CONS(_five)(_one)))(_five))\
  (IS_EQUAL(CAR(CONS(_two)(CONS(_one)(_three))))(_two))))

TEST('CDR')(ASSERT(AND\
  (IS_EQUAL(CDR(CONS(_five)(_one)))(_one))\
  (IS_EQUAL(CAR(CDR(CONS(_two)(CONS(_one)(_three)))))(_one))))

TEST('CONS')(ASSERT(AND\
  (IS_EQUAL(CDR(CDR(CONS(_two)(CONS(_one)(_three)))))(_three))\
  (IS_EQUAL(CAR(CDR(CONS(_five)(CONS(_two)(CONS(_one)(_three))))))(_two))))

TEST('RANGE')(ASSERT(AND(\
  AND\
    (IS_EQUAL(CAR(RANGE(_three)(_five)))(_three))\
    (IS_EQUAL(CAR(CDR(RANGE(_three)(_five))))(_four)))(\
  AND\
    (IS_EQUAL(CAR(CDR(CDR(RANGE(_three)(_five)))))(_five))\
    (IS_NULL(CDR(CDR(CDR(RANGE(_three)(_five)))))))))

TEST('MAP')(ASSERT(AND(\
  AND\
    (IS_EQUAL\
      (CAR(MAP(RANGE(_three)(_five))(lambda v:POWER(v)(_two))))\
      (POWER(_three)(_two)))\
    (IS_EQUAL\
      (CAR(CDR(MAP(RANGE(_three)(_five))(lambda v:POWER(v)(_two)))))\
      (POWER(_four)(_two))))(\
  AND\
    (IS_EQUAL\
       (CAR(CDR(CDR(MAP(RANGE(_three)(_five))(lambda v:POWER(v)(_two))))))\
       (POWER(_five)(_two)))\
    (IS_NULL(CDR(CDR(CDR(MAP(RANGE(_three)(_five))(lambda v:POWER(v)(_two))))))))))

# Examples
print('\n--- Examples ---\n')

# 階層 FACTORIAL(n) = n!
FACTORIAL = Y(lambda f:lambda n:IF(IS_ZERO(n))\
  (lambda _:SUCCESSOR(n))\
  (lambda _:MULTIPLICATION(n)(f(PREDECESSOR(n))))\
(NIL))

# 費氏數列函數 FIBONACCI(n)
FIBONACCI = Y(lambda f:lambda n:\
  IF(IS_LESS_THAN_EQUAL(n)(SUCCESSOR(lambda f:IDENTITY)))\
  (lambda _:n)\
  (lambda _:ADDITION\
    (f(PREDECESSOR(n)))\
    (f(PREDECESSOR(PREDECESSOR(n)))))\
(NIL))

TEST('FACTORIAL: 5! = 120')(ASSERT(IS_EQUAL\
  (FACTORIAL(_five))\
  (ADDITION(MULTIPLICATION(_ten)(_ten))(ADDITION(_ten)(_ten)))))

TEST('FIBONACCI: 10 = 55')(ASSERT(IS_EQUAL\
  (FIBONACCI(_ten))\
  (ADDITION(MULTIPLICATION(_five)(_ten))(_five))))
