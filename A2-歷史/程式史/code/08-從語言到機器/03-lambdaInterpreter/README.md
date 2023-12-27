# Nameless!

A lambda calculus interpreter written in Python. Lambda calculus is a
minimal, turing complete programming language. In lambda calculus,
everything is an anonymous (i.e. nameless :wink:) function. This
project has no dependencies outside of the standard library.

## Execution

1. Clone this repository:

    ```
    git clone https://github.com/ElliotPenson/nameless.git
    ```

2. Move into the newly created directory:

    ```
    cd nameless
    ```

3. Run the `nameless` package:

    ```
    python nameless
    ```

## Grammar

```
Expression  -> Variable | Application | Abstraction
Variable    -> ID
Application -> (Expression Expression)
Abstraction -> λID.Expression
```

An **abstraction** `λx.e` is a definition of an anonymous function
capable of taking a single input `x` and substituting it into (and
thus evaluating) the expression `e`.  An **application** `(e1 e2)`
applies the `e1` function to the `e2` value, that is, it represents
the act of calling function `e1` on input `e2`.

Note that the at-sign (`@`) may be used instead of lambda (`λ`) to
denote an abstraction.

## Features

Execution follows *normal order reduction*. This means that the
outmost lambda expression is always applied first. The interpreter
displays each reduction step on a different line:

```
> (λm.((m λt.λf.t) λx.λt.λf.f) λz.λs.z)
((λz.λs.z λt.λf.t) λx.λt.λf.f)
(λs.λt.λf.t λx.λt.λf.f)
λt.λf.t
```

Syntax errors are nicely displayed to user:

```
> λx
ParseError: Expected: ., Found: EOF
```

During α-conversion, the interpreter has the capability to rename
variables to avoid variable capture:

```
> (λx.λy.(x y) y)
λa.(y a)
```

## Classes

* `Lexer`
  * Found in the `lexer` module.
  * An iterator that splits source code into tokens.
* `Parser`
  * Found in the `parser` module.
  * An LL(1) parser that performs syntactic analysis on lambda
    calculus source code. An abstract syntax tree is provided if the
    given expression is valid. Any issues with the syntax cause a
    ParserError to be raised.
* `Variable`, `Application`, `Abstraction`
  * Found in the `lambda_calculus_ast` module.
  * Encapsulates each lambda calculus nonterminal (see CFG above). All
    three classes are subclasses of `Expression`.
* `FreeVariables`, `BoundVariables`
  * Found in the `visitors` module.
  * Visitors that traverse a lambda calculus abstract syntax tree and
    return a set of all free or bound variables.
* `AlphaConversion`
  * Found in the `visitors` module.
  * Nondestructively substitutes all free occurrences of a particular
    variable for an arbitrary expression.
* `BetaReduction`
  * Found in the `visitors` module.
  * Embodies the act of applying a function to a parameter. Provides a
    new abstract syntax tree with a single reduction performed (if
    possible).
