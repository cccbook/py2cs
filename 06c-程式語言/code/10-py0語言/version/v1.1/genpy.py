"""
STMTS = stmt*
STMT = BLOCK | FUNC | IF | WHILE | RETURN | ASSIGN | CALL
IF = if expr: stmt (elif stmt)* (else stmt)?
WHILE = while expr: stmt
"""

def gen(n):
    t = n['type']
    match t:
        case 'stmts':
            for stmt in n['stmts']:
                gen(stmt)
        case 'func':
            
