#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
lexer.py

@author ejnp
"""

import string

from collections import namedtuple

PUNCTUATION = [u'λ', '@', '.', '(', ')']
WHITESPACE = list(string.whitespace)

Token = namedtuple('Token', ['type', 'value'])


class Lexer(object):
    """An iterator that splits lambda calculus source code into Tokens.

    Attributes:
        source (str): Lambda calculus source code
        size (int): The number of characters in the source
        position (int): Current index in the source
    """

    def __init__(self, source):
        self.source = source
        self.size = len(source)
        self.position = 0

    def __iter__(self):
        return self

    def next(self):
        """Returns the next lexeme as a Token object."""
        self._clear_whitespace()
        if self.position > self.size:
            raise StopIteration()
        elif self.position == self.size:
            self.position += 1
            return Token('EOF', None)
        elif self.source[self.position] in PUNCTUATION:
            char = self.source[self.position]
            self.position += 1
            return Token(char, None)
        else:
            symbol = ''
            while (self.position < self.size and
                   not self.source[self.position] in PUNCTUATION + WHITESPACE):
                symbol += self.source[self.position]
                self.position += 1
            return Token('SYMBOL', symbol)

    def _clear_whitespace(self):
        """Advances position past any whitespace."""
        while (self.position < self.size and
               self.source[self.position] in string.whitespace):
            self.position += 1
