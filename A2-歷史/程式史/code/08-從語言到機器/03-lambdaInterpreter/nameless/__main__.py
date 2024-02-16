#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
main.py

@author ejnp
"""

from lexer import Lexer
from parser import Parser, ParserError
from visitors import BetaReduction


def interpret(input_string, print_reductions=False):
    """Performs normal order reduction on the given string lambda calculus
    expression. Returns the expression's normal form if it exists.
    """
    lexer = Lexer(input_string)
    try:
        ast = Parser(lexer).parse()
    except ParserError as discrepancy:
        print('ParseError: ' + discrepancy.message)
        return None
    normal_form = False
    while not normal_form:
        reducer = BetaReduction()
        reduced_ast = reducer.visit(ast)
        normal_form = not reducer.reduced
        if print_reductions:
            print(ast) # print(unicode(ast))
        ast = reduced_ast
    return ast # unicode(ast)


def main():
    """Begins an interactive lambda calculus interpreter"""
    print("nameless!\nType 'quit' to exit.")
    while True:
        read = input('> ') # raw_input('> ').decode('utf-8')
        if read == 'quit':
            break
        if read != '':
            interpret(read, print_reductions=True)


if __name__ == '__main__':
    main()
