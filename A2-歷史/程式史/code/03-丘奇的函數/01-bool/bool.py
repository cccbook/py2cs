TRUE  = lambda t:lambda f:t
FALSE = lambda t:lambda f:f
IF  = lambda c:lambda t:lambda f:c(t)(f)

print('IF(TRUE)("YES")("NO")=', IF(TRUE)("YES")("NO"))
print('IF(FALSE)("YES")("NO")=', IF(FALSE)("YES")("NO"))
