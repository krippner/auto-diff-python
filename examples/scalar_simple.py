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

# Compute the gradient of f using reverse-mode AD
f.pull_gradient_at(z)

# Get the components of the gradient
print("∂z/∂x =", d(x))       # ∂z/∂x = -2.0
print("∂z/∂y =", d(y))       # ∂z/∂y = 1.5
