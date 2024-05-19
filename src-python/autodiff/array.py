"""
AutoDiff for NumPy arrays
=========================

This module provides automatic differentiation for scalar and
(1D and 2D) NumPy array computations.

Core classes
------------
Function
    Lets you evaluate and differentiate a program defined by
    variables and expressions.

Variable classes
----------------
ScalarVariable
    A variable storing `float` value and
    `np.ndarray[np.float64[m, n]]` derivative.
VectorVariable
    A variable storing `np.ndarray[np.float64[r, 1]]` value
    and `np.ndarray[np.float64[m, n]]` derivative.
MatrixVariable
    A variable storing `np.ndarray[np.float64[r, s]]` value
    and `np.ndarray[np.float64[m, n]]` derivative.

Operations
----------
In binary operations, one of the operands can also be a scalar
or array literal.

>>> x = var(np.array([1., 2., 3.]))  # vector variable

>>> u = x + np.array([4., 5., 6.])   # add array literal

Scalar literals and expressions are broadcasted to the shape
of the array.

>>> x = var(np.array([[1., 2.], [3., 4.]]))  # matrix variable

>>> u = x + 5   # add 5 to each element

+, -, *, /, **
    Element-wise arithmetic operations.
sin
    Sine function, element-wise.
cos
    Cosine function, element-wise.
exp
    Exponential function, element-wise.
log
    Natural logarithm, element-wise.
sqrt
    Square root, element-wise.
square
    Square, element-wise.
minimum
    Element-wise minimum of an expression and zero.
maximum
    Element-wise maximum of an expression and zero.
dot
    Dot product of two vectors.
outer
    Outer (tensor) product of two vectors.
matmul, @
    Matrix multiplication.
sum
    Sum of array expression.
mean
    Arithmetic mean of array expression.
norm
    Frobenius (L²) norm of array expression.
squared_norm
    Squared Frobenius (L²) norm of array expression.

Matrix-valued expressions
-------------------------
During differentiation, AutoDiff flattens matrix expressions
in column-major order.
This ensures that the derivative (Jacobian matrix) is always
a 2D NumPy array.

>>> m1 = np.array([[1., 2.], [3., 4.]])

>>> m2 = np.array([[5., 6., 7.], [8., 9., 10.]])

>>> x = var(m1)    # 2⨉2 matrix variable

>>> y = var(m2)    # 2⨉3 matrix variable

>>> u = var(x @ y) # 2⨉3 matrix variable

>>> f = Function(u)

>>> f.pull_gradient_at(u)

>>> d(u)           # 6⨉6 identity matrix

>>> d(x)           # 6⨉4 matrix

>>> d(y)           # 6⨉6 matrix

1D arrays vs vectors 
--------------------

1D arrays and N⨉1 arrays (columns) in NumPy are treated as vectors
in AutoDiff. But 1⨉N arrays are treated as matrices.
"""

from autodiff._array import __version__
from autodiff._array import *

__all__ = [
    "Function",
    "Variable",
    "var",
    "d",
    "ScalarExpression",
    "ScalarOperation",
    "ScalarVariable",
    "VectorExpression",
    "VectorOperation",
    "VectorVariable",
    "MatrixExpression",
    "MatrixOperation",
    "MatrixVariable",
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
    "square",
    "minimum",
    "maximum"
    "dot",
    "outer",
    "matmul",
    "sum",
    "mean",
    "norm",
    "squared_norm",
]
