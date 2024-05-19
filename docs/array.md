# The `autodiff.array` module

The `autodiff.array` submodule provides automatic differentiation for scalar and (1D and 2D) NumPy array computations.

## Classes

Core class | Description
--- | ---
`class Variable` | Base class for all variables. Do not use directly.
`class Function` | Lets you evaluate and differentiate a program defined by variables and expressions.

Expression class | Description
--- | ---
`class ScalarExpression` | Base class for all scalar expressions. Do not use directly.
`class VectorExpression` | Base class for all 1D array expressions. Do not use directly.
`class MatrixExpression` | Base class for all 2D array expressions. Do not use directly.

Variable class | Value type | Derivative type
--- | --- | ---
`class ScalarVariable(ScalarExpression, Variable)` | `float` | `np.ndarray[np.float64[m, n]]`
`class VectorVariable(VectorExpression, Variable)` | `np.ndarray[np.float64[r, 1]]` | `np.ndarray[np.float64[m, n]]`
`class MatrixVariable(MatrixExpression, Variable)` | `np.ndarray[np.float64[r, s]]` | `np.ndarray[np.float64[m, n]]`

## Variable factory functions

The following table assumes that `scalar_expr` has base type `ScalarExpression`, `vector_expr` has base type `VectorExpression`, and `matrix_expr` has base type `MatrixExpression`.

Function call | Return type | Comment
--- | --- | ---
`var(1.0)` | `ScalarVariable` |
`var(scalar_expr)` | `ScalarVariable` |
`var(np.array([1, 2, 3]))` | `VectorVariable` |
`var(np.array([[1], [2], [3]]))` | `VectorVariable` | N⨉1 arrays (columns) are treated as vectors.
`var(vector_expr)` | `VectorVariable` |
`var(np.array([[1, 2, 3]]))` | `MatrixVariable` | 1⨉N arrays (rows) are treated as matrices.
`var(np.array([[1, 2], [3, 4]]))` | `MatrixVariable` |
`var(matrix_expr)` | `MatrixVariable` |

## Operations

In binary operations, one of the operands can also be a scalar or array literal.

```python
x = var(np.array([1., 2., 3.]))  # vector variable
x + np.array([4., 5., 6.])       # add array literal
```

Scalar literals and expressions are broadcasted to the shape of the array.

```python
x = var(np.array([[1., 2.], [3., 4.]]))  # matrix variable
x + 5  # add 5 to each element
```

The following operations are currently supported:

- `+`, `-`, `*`, `/`, `**`: Element-wise arithmetic operations.
- `sin`: Sine function, element-wise.
- `cos`: Cosine function, element-wise.
- `exp`: Exponential function, element-wise.
- `log`: Natural logarithm, element-wise.
- `sqrt`: Square root, element-wise.
- `square`: Square function, element-wise.
- `minimum`: Element-wise minimum of an expression and zero.
- `maximum`: Element-wise maximum of an expression and zero.
- `dot`: Dot product of two vectors.
- `outer`: Outer (tensor) product of two vectors.
- `matmul`, `@`: Matrix-matrix or matrix-vector product.
- `sum`: Sum of array expression.
- `mean`: Arithmetic mean of array expression.
- `norm`: Frobenius ($L^2$) norm of array expression.
- `squared_norm`: Squared Frobenius ($L^2$) norm of array expression.

## Matrix-valued expressions

During differentiation, AutoDiff flattens matrix expressions in column-major order.
This ensures that the derivative (Jacobian matrix) is always a 2D NumPy array.

```python
m1 = np.array([[1., 2.], [3., 4.]])
m2 = np.array([[5., 6., 7.], [8., 9., 10.]])
x = var(m1)    # 2⨉2 matrix variable
y = var(m2)    # 2⨉3 matrix variable
u = var(x @ y) # 2⨉3 matrix variable
f = Function(u)
f.pull_gradient_at(u)
d(u)           # 6⨉6 identity matrix
d(x)           # 6⨉4 matrix
d(y)           # 6⨉6 matrix
```
