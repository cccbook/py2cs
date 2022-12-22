TRUE  = lambda t:lambda f:t
FALSE = lambda t:lambda f:f
IF  = lambda c:lambda t:lambda f:c(t)(f)

print('IF(TRUE)("TRUE")("FALSE")=', IF(TRUE)("TRUE")("FALSE"))
print('IF(FALSE)("TRUE")("FALSE")=', IF(FALSE)("TRUE")("FALSE"))
