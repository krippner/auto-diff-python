# AutoDiff - automatic differentiation for Python

[![Wheel](https://github.com/krippner/auto-diff-python/actions/workflows/wheel.yml/badge.svg)](https://github.com/krippner/auto-diff-python/actions/workflows/wheel.yml)

A lightweight Python package that provides fast **automatic differentiation (AD)** in forward and reverse mode for scalar and array computations.

AD is an **efficient algorithm** for computing **exact derivatives** of numeric programs.
It is a standard tool in numerous fields, including optimization, machine learning, and scientific computing.

> [!NOTE]
> This repository focuses on providing Python bindings and does not include the C++ backend, which is part of a separate, standalone C++ library. The C++ version offers additional features not available in these bindings. For more information on the C++ version, please visit the [AutoDiff repository](https://github.com/krippner/auto-diff).

## Features

- **Automatic differentiation**:
  - Jacobian matrix with forward- and reverse-mode AD
  - Jacobian-vector products, e.g., gradients and directional derivatives
  - Support for scalar, 1D and 2D array, and linear algebra computations
- **Fast and efficient implementation**:
  - Backend written in C++ (using [this repository](https://github.com/krippner/auto-diff))
  - Leverages the performance of the [Eigen](https://eigen.tuxfamily.org) linear algebra library
  - Memory efficient (see [Variables vs. expressions](docs/expressions.md#variables-vs-expressions))
- **Simple and intuitive API**
  - Regular control flow: function calls, loops, branches
  - Eager evaluation: what you evaluate is what you differentiate
  - Lazy re-evaluations: offering you precise control over what to evaluate
  - Math-inspired syntax

For more details, see the [documentation](#documentation).

## Installation

If you are on Linux, you can download the latest wheel file from the [releases page](https://github.com/krippner/auto-diff-python/releases) and install it using pip.

```bash
python -m pip install autodiff-0.1.0-cp311-cp311-linux_x86_64.whl
```

The wheel contains the extension modules as well as Python stub files for autocompletion and documentation in your IDE.

## Usage

Below are two simple examples of how to use the `autodiff` package to compute the gradient of a function with respect to its inputs.

The package provides two sub-modules: `array` and `scalar`.
The `array` module is more general and can be used to compute gradients of functions involving both scalars and arrays (1D and 2D).

> [!CAUTION]
> It is not possible to mix the `array` and `scalar` modules in the same program, as they use incompatible internal representations for variables.

### NumPy array example

```python
# Example: gradient computation with NumPy arrays
import numpy as np
from autodiff.array import Function, var, d

# Create two 1D array variables
x = var(np.array([1, 2, 3]))
y = var(np.array([4, 5, 6]))

# Assign their (element-wise) product to a new variable
z = var(x * y)

# Variables are evaluated eagerly
print("z =", z())            # z = [ 4. 10. 18.]

# Create the function f : (x, y) ↦ z = x * y
f = Function(z) # short for: Function(sources=(x, y), target=z)

# Set the (element-wise) derivative of z with respect to itself
z.set_derivative(np.ones((1, 3)))

# Compute the gradient of f at (x, y) using reverse-mode AD
f.pull_gradient()

# Get the components of the (element-wise) gradient
print("∇_x f =", d(x))       # ∇_x f = [[4. 5. 6.]]
print("∇_y f =", d(y))       # ∇_y f = [[1. 2. 3.]]

```

### Scalar example

For functions mapping only scalars to scalars, the `scalar` module is more efficient and convenient.
No further imports are required.

```python
# Example: gradient computation with scalars
from autodiff.scalar import Function, var, d

# Create two scalar variables
x = var(1.5)
y = var(-2.0)

# Assign their product to a new variable
z = var(x * y)

# Variables are evaluated eagerly
print("z =", z())            # z = -3.0

# Create the function f : (x, y) ↦ z = x * y
f = Function(z) # short for: Function(sources=(x, y), target=z)

# Compute the gradient of f at (x, y) using reverse-mode AD
f.pull_gradient_at(z)

# Get the components of the gradient
print("∂f/∂x =", d(x))       # ∂f/∂x = -2.0
print("∂f/∂y =", d(y))       # ∂f/∂y = 1.5

```

## Documentation

1. [Variables and expressions](docs/expressions.md#top) - writing programs with `autodiff`
   1. [Variables](docs/expressions.md#variables)
   2. [Expressions](docs/expressions.md#expressions)
   3. [Variables vs. expressions](docs/expressions.md#variables-vs-expressions)
2. [Functions](docs/functions.md#top) - (deferred) evaluation and differentiation
   1. [Lazy evaluation](docs/functions.md#lazy-evaluation)
   2. [Forward-mode differentiation](docs/functions.md#forward-mode-differentiation)
   3. [Reverse-mode differentiation (aka backpropagation)](docs/functions.md#reverse-mode-differentiation-aka-backpropagation)
   4. [Advanced: changing the program after evaluation](docs/functions.md#advanced-changing-the-program-after-evaluation)
3. [The `autodiff.scalar` module](docs/scalar.md#top) - working with scalars only
   1. [Classes](docs/scalar.md#classes)
   2. [Variable factory functions](docs/scalar.md#variable-factory-functions)
   3. [Operations](docs/scalar.md#operations)
4. [The `autodiff.array` module](docs/array.md#top) - working with scalars and NumPy arrays
   1. [Classes](docs/array.md#classes)
   2. [Variable factory functions](docs/array.md#variable-factory-functions)
   3. [Operations](docs/array.md#operations)
   4. [Matrix-valued expressions](docs/array.md#matrix-valued-expressions)
5. [Applications](docs/applications.md#top) - common use cases and examples
   1. [Control flow](docs/applications.md#control-flow)
   2. [Computing the Jacobian matrix](docs/applications.md#computing-the-jacobian-matrix)
   3. [Gradient computation](docs/applications.md#gradient-computation)
   4. [Element-wise gradient computation](docs/applications.md#element-wise-gradient-computation)
   5. [Jacobian-vector products](docs/applications.md#jacobian-vector-products)
