# Example: gradient computation with NumPy arrays
import numpy as np
from autodiff.array import Function, var, d

# Create two vector variables
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

# Compute the gradient of f using reverse-mode AD
f.pull_gradient()

# Get the components of the (element-wise) gradient
print("∂z/∂x =", d(x))       # ∂z/∂x = [[4. 5. 6.]]
print("∂z/∂y =", d(y))       # ∂z/∂y = [[1. 2. 3.]]
