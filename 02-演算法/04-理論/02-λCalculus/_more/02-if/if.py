TRUE  = lambda t:lambda f:t
FALSE = lambda t:lambda f:f
NOT   = lambda c:c(FALSE)(TRUE)
IF    = lambda c:lambda t:lambda f:c(t)(f)

ASSERT = lambda truth: (IF(truth)
    (lambda description:f'[✓] ${description}')
    (lambda description:f'[✗] ${description}')
)

REFUTE = lambda truth:ASSERT(NOT(truth))

TEST   = lambda description:lambda assertion:print(assertion(description))

TEST('TRUE')(ASSERT(TRUE))

TEST('FALSE')(REFUTE(FALSE))
