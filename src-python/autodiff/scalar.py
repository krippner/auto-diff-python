"""
AutoDiff for scalars
====================

This module provides automatic differentiation for scalar computations.

Core classes
------------
Function
    Lets you evaluate and differentiate a program defined by
    variables and expressions.

Variable classes
----------------
ScalarVariable
    A variable storing `float` value and `float` derivative.

Operations
----------
In binary operations, one of the operands can also be a scalar literal.

>>> x = var(2)  # scalar variable

>>> u = x + 3   # add scalar literal

+, -, *, /, **
    Arithmetic operations.
sin
    Sine function.
cos
    Cosine function.
exp
    Exponential function.
log
    Natural logarithm.
sqrt
    Square root.
square
    Square function.
minimum
    Minimum of a scalar expression and zero.
maximum
    Maximum of a scalar expression and zero.
"""

from autodiff._scalar import __version__
from autodiff._scalar import *

__all__ = [
    "Function",
    "Variable",
    "var",
    "d",
    "ScalarExpression",
    "ScalarOperation",
    "ScalarVariable",
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
    "square",
    "minimum",
    "maximum",
]
