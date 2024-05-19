# The `autodiff.scalar` module

The `autodiff.scalar` submodule provides a simple and efficient way to compute derivatives of functions mapping scalars to scalars.

## Classes

Core class | Description
--- | ---
`class Variable` | Base class for all variables. Do not use directly.
`class Function` | Lets you evaluate and differentiate a program defined by variables and expressions.

Expression class | Description
--- | ---
`class ScalarExpression` | Base class for all scalar expressions. Do not use directly.

Variable class | Value type | Derivative type
--- | --- | ---
`class ScalarVariable(ScalarExpression, Variable)` | `float` | `float`

## Variable factory functions

The following table assumes that `scalar_expr` has base type `ScalarExpression`.

Function call | Return type
--- | ---
`var(1.0)` | `ScalarVariable`
`var(scalar_expr)` | `ScalarVariable`

## Operations

In binary operations, one of the operands can also be a scalar literal.

```python
x = var(2)  # scalar variable
x + 3       # add scalar literal
```

The following operations are currently supported:

- `+`, `-`, `*`, `/`, `**`: Arithmetic operations.
- `sin`: Sine function.
- `cos`: Cosine function.
- `exp`: Exponential function.
- `log`: Natural logarithm.
- `sqrt`: Square root.
- `square`: Square function.
- `minimum`: Minimum of a scalar expression and zero.
- `maximum`: Maximum of a scalar expression and zero.
