#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
lexer.py

@author ejnp
"""


class Expression(object):
    """Abstract class for any lambda calculus expression."""

    def children(self):
        """Returns a list of Expression objects."""
        pass


class Variable(Expression):
    """Encapsulates a lambda calculus variable.

    Attributes:
        name (str): The variable's ID
    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def children(self):
        return []


class Application(Expression):
    """Encapsulates a lambda calculus function call.

    Attributes:
        left_expression (Expression): A function to be evaluated
        right_expression (Expression): The argument that's applied
    """

    def __init__(self, left_expression, right_expression):
        self.left_expression = left_expression
        self.right_expression = right_expression

    def __str__(self):
        return u'({} {})'.format(self.left_expression, self.right_expression)

    def children(self):
        return [self.left_expression, self.right_expression]


class Abstraction(Expression):
    """Encapsulates a function in lambda calculus.

    Attributes:
        parameter (Variable): The argument variable
        body (Expression): The scope of the function
    """

    def __init__(self, parameter, body):
        self.parameter = parameter
        self.body = body

    def __str__(self):
        return u'λ{}.{}'.format(self.parameter, self.body)

    def children(self):
        return [self.parameter, self.body]
