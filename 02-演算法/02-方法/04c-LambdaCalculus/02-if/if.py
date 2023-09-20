TRUE  = lambda t:lambda f:t
FALSE = lambda t:lambda f:f
NOT   = lambda c:c(FALSE)(TRUE)
IF    = lambda c:lambda t:lambda f:c(t)(f)

ASSERT = lambda truth: (IF(truth)
    (lambda description:f'[\x1b[32m✓\x1b[0m] ${description}')
    (lambda description:f'[\x1b[31m✗\x1b[0m] ${description}')
)

REFUTE = lambda truth:ASSERT(NOT(truth))

TEST   = lambda description:lambda assertion:print(assertion(description))

TEST('TRUE')(ASSERT(TRUE))

TEST('FALSE')(REFUTE(FALSE))
