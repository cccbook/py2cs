# Church Booleans
TRUE  = lambda t:lambda f:t # true: λ t f. t
FALSE = lambda t:lambda f:f # false: λ t f. f
AND   = lambda p:lambda q:p(q)(p) # ??
OR    = lambda p:lambda q:p(p)(q) # ??
XOR   = lambda p:lambda q:p(NOT(q))(q) # ??
NOT   = lambda c:c(FALSE)(TRUE) # (cond?FALSE:TRUE)
IF    = lambda c:lambda t:lambda f:c(t)(f) # if: λ p x y. p x y # if p then x else y.
# 範例 IF(TRUE)(FALSE)(TRUE) => TRUE(FALSE)(TRUE) 

ASSERT = lambda truth: (IF(truth)\
    (lambda description:f'[\x1b[32m✓\x1b[0m] ${description}')\
    (lambda description:f'[\x1b[31m✗\x1b[0m] ${description}')
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

