# py0 -- EBNF

```
STMTS  = STMT*                                            # list
STMT   = BLOCK | FUNC | IF | WHILE | RETURN | ASSIGN | CALL
IF     = if EXPR: STMT (elif STMT)* (else STMT)?
WHILE  = while EXPR: STMT
FOR    = for id in EXPR: STMT
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
TERM   = OBJ ( [EXPR] | . id | (ARGS) )*
OBJ    = id | string | int | float | LREXPR
CALL   = id(ARGS)
ARGS   = (EXPR ,)* EXPR?
ARRAY  = [ (EXPR ,)* EXPR? ]
MAP    = { (PAIR ,)* PAIR? }
PAIR   = string : EXPR
```