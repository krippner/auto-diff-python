# Applications

This section provides a list of examples demonstrating the use `autodiff` in common scenarios.
The examples assume that you imported the package as follows:

```python
from autodiff.array import Function, var, d
```

## Control flow

### Function calls

```python
# This function can take any combination of
# literals and variables (both scalar and array).
def logistic(x, k):
    return 1 / (1 + exp(-k * x))  # scalars broadcast to arrays

k = var(4)
y = var(logistic(0, k))
print("y =", y())  # y = 0.5

x = var(np.array([-1, 0, 1]))
y = var(logistic(x, 4))
print("y =", y())  # y = [0.01798621 0.5        0.98201379]

k = var(np.array([4, 4, 4]))
y = var(logistic(x, k))
print("y =", y())  # y = [0.01798621 0.5        0.98201379]
```

### Loops

```python
initial = var(0)
state = initial
for i in range(10):
    state = var(state + 1)   # evaluate to a NEW variable

print("state =", state())    # state = 10.0

f = Function(state)
f.pull_gradient_at(state)
print("∂state/∂initial =", d(initial)) # ∂state/∂initial = 1.0
```

For more details on when to use `var`, see [Variables vs expressions](expressions.md#variables-vs-expressions).

### Branches

```python
# Caution: if statements are not differentiable
def bad_relu(x):
    return x if x > 0 else 0 # cannot differentiate this

def good_relu(x):
    return maximum(x)

x = var(np.array([-1, 0, 1]))
y = var(good_relu(x))
print("y =", y())            # y = [0. 0. 1.]
```

You can, however, use conditionals to decide which expression to evaluate:

```python
# Caution: if statements cannot depend on variables
x = var(1)
if x() > 0:        # true
    y = var(x)
else:
    y = var(0)     # never evaluated!
# from now on y = var(x)

f = Function(y)
x.set(-1)
f.evaluate()
print("y =", y())  # y = -1.0
```

## Computing the Jacobian matrix

Given a function $f \colon \mathbb{R}^m \to \mathbb{R}^n$, $x \mapsto y$, the Jacobian matrix $J_f(x) \in \mathbb{R}^{n \times m}$ is defined as

$$
J_f(x) = \begin{bmatrix}
    \frac{\partial f_1}{\partial x_1} \ \ldots\ \frac{\partial f_1}{\partial x_m} \\
    \vdots \\
    \frac{\partial f_n}{\partial x_1} \ \ldots\ \frac{\partial f_n}{\partial x_m}
\end{bmatrix} .
$$

```python
x = var(np.array([1, 2, 3]))
m = var(np.array([[1, 2, 3], [4, 5, 6]]))
y = var(m @ x)            # matrix-vector product

f = Function(y)
f.pull_gradient_at(y)
print("∂f/∂x =\n", d(x))  # ∂f/∂x =
                          #  [[1. 2. 3.]
                          #   [4. 5. 6.]]
print("∂f/∂m =\n", d(m))  # ∂f/∂m =
                          #  [[1. 0. 2. 0. 3. 0.]
                          #   [0. 1. 0. 2. 0. 3.]]
```

> [!NOTE]
> During differentiation, matrices are flattened column-wise.
> Therefore, `d(m)` returns a $2 \times 6$ Jacobian matrix instead of a $2 \times 2 \times 3$ tensor.
> For more details, see [Matrix-valued expressions](array.md#matrix-valued-expressions).

For more details on the `pull_gradient_at` method, see [Reverse-mode differentiation (aka backpropagation)](functions.md#reverse-mode-differentiation-aka-backpropagation).

## Gradient computation

Given a scalar function $f \colon \mathbb{R}^m \to \mathbb{R}$, $x \mapsto y$, the gradient $\nabla f(x) \in \mathbb{R}^m$ is defined as

$$
\nabla f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots \right] .
$$

```python
x = var(np.array([1, 2, 3]))
y = var(norm(x))       # L²-norm

f = Function(y)        # f : R³ → R, x ↦ y = ||x||
f.pull_gradient_at(y)
print("∇f =", d(x))    # ∇f = [[0.26726124 0.53452248 0.80178373]]
```

Note that the gradient of a scalar function is a $1 \times n$ Jacobian matrix (aka "row vector").

For more details, see [Reverse-mode differentiation (aka backpropagation)](functions.md#reverse-mode-differentiation-aka-backpropagation).

## Element-wise gradient computation

```python
x = var(np.array([1, 2, 3]))
y = var(np.array([4, 5, 6]))
z = var(x * y)          # element-wise multiplication

f = Function(z)
z.set_derivative(np.ones((1, 3))) # element-wise identity
f.pull_gradient()
print("∇_x f =", d(x))  # ∇_x f = [[4. 5. 6.]]
print("∇_y f =", d(y))  # ∇_y f = [[1. 2. 3.]]
```

If you used `pull_gradient_at(z)` (as in the previous example), you would instead get $3 \times 3$ diagonal matrices for $\nabla_x f$ and $\nabla_y f$ (with the same values).

## Jacobian-vector products

Given a function $f \colon \mathbb{R}^m \to \mathbb{R}^n$, $x \mapsto y$.
If you are only interested in the product of the Jacobian matrix $J_f(x)$ with a given direction (tangent vector) $\delta x$, 

$$
\delta y = J_f \cdot \delta x \ ,
$$

then the following code using the `push_tangent` method is much more efficient than computing the full Jacobian matrix first and then multiplying it with $\delta x$.

```python
# Directional derivative (Jacobian-vector product)
x = var(np.array([1, 2, 3]))
m = np.array([[1, 2, 3], [4, 5, 6]])
y = var(matmul(m, x))     # matrix-vector product

f = Function(y)           # f : R³ → R², x ↦ y = m @ x
δx = np.array([1, 1, 1])  # direction vector
x.set_derivative(δx)
f.push_tangent()
print("δy =\n", d(y))     # δy =
                          # [[ 6.]
                          #  [15.]]
```

For more details, see [Forward-mode differentiation](functions.md#forward-mode-differentiation).
