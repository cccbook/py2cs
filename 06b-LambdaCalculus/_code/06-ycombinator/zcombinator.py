# from -- https://gist.github.com/ishikawa/984200

# Y Combinator in Python
# See <http://d.hatena.ne.jp/nowokay/20090409#1239268405>

true  = lambda x: lambda y: x # if true do x  兩個參數執行第一個
false = lambda x: lambda y: y # if false do y 兩個參數執行第二個

# `bool` to `true|false`
#
# >>> boolean(3 < 4)(2)(5)
# 2
# >>> boolean(3 < 1)(2)(5)
# 5
IF = lambda b: true if b else false

print(IF(3 < 4)(2)(5))
# Y Combinator
#
# >>> Y(lambda f: lambda n: n if n < 2 else f(n-1) + f(n-2))(7)
# Traceback (most recent call last):
#     ...
# RuntimeError: maximum recursion depth exceeded
Y = lambda f: (lambda x: f(x(x)))\
              (lambda x: f(x(x)))
# print(Y(lambda f: lambda n: n if n < 2 else f(n-1) + f(n-2))(7))

# Z Combinator
#
# Example: A recursive function to compute fibonacci numbers.
#
# >>> from * import ycombinator
# >>> Z(lambda f: lambda n: n if n < 2 else f(n-1) + f(n-2))(7)
# 13
# >>> Z(lambda f: lambda n: boolean(n < 2)(n)(f(n-1) + f(n-2)))(7)
# Traceback (most recent call last):
#     ...
# RuntimeError: maximum recursion depth exceeded
# >>> Z(lambda f: lambda n: boolean(n < 2)(lambda: n)(lambda: f(n-1) + f(n-2))())(7)
# 13
Z = lambda f: (lambda x: (lambda m: f(x(x))(m)))\
              (lambda x: (lambda m: f(x(x))(m)))

print(Z(lambda f: lambda n: \
  IF(n < 2)(lambda: n)(lambda: f(n-1) + f(n-2))())(7))
# 7 傳給了 n 包進去變 closure，然後 () 會觸發執行 IF 傳回的結果
# 也就是 (lambda n 或 lambda: f(n-1)+f(n-2))()。 
