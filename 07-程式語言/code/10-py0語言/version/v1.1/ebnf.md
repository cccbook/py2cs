# py0 -- EBNF

```
STMTS  = stmt*                                            # list
STMT   = BLOCK | FUNC | IF | WHILE | RETURN | ASSIGN | CALL
IF     = if expr: stmt (elif stmt)* (else stmt)?   # expr1, stmt1, stmts2, stmt3
WHILE  = while expr: stmt
RETURN = return expr
ASSIGN = id = expr
CALL   = id(ARGS)
FUNC   = def id(PARAMS): BLOCK
PARAMS = PARAM*                                           # list
PARAM  = id
BLOCK  = begin STMTS end
EXPR   = BEXPR (if EXPR else EXPR)?
BEXPR  = CEXPR ((and|or) CEXPR)*                           # list
CEXPR  = MEXPR (['==', '!=', '<=', '>=', '<', '>'] MEXPR)* # list
MEXPR  = ITEM (['+', '-', '*', '/', '%'] ITEM)*            # list
ITEM   = FACTOR
FACTOR = float | integer | ( expr ) | TERM
TERM   = id | CALL
CALL   = id(ARGS)
ARGS   = (EXPR ',')* EXPR?
```