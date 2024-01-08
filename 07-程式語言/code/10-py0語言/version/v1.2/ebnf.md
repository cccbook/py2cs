# py0 -- EBNF

```
STMTS  = STMT*                                            # list
STMT   = BLOCK | FUNC | IF | WHILE | RETURN | ASSIGN | CALL
IF     = if EXPR: STMT (elif STMT)* (else STMT)?   # expr1, stmt1, stmts2, stmt3
WHILE  = while EXPR: STMT
RETURN = return EXPR
ASSIGN = id = EXPR
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
FACTOR = float | integer | LREXPR | TERM
LREXPR = ( EXPR )
TERM   = id | CALL
CALL   = id(ARGS)
ARGS   = (EXPR ',')* EXPR?
```