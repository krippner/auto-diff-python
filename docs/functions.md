# Functions

A *function* allows you to lazily evaluate and differentiate a program.
A program is a collection of variables and expressions, and the corresponding function is created by specifying the target variable(s) and, if needed, the source variables.

```python
# Creating a function with a single target variable
x = var(1)
y = var(2)
u = var(x * y)
f = Function(sources=(x, y), target=u)

# Creating a function with multiple target variables
v = var(x + y)
g = Function(sources=(x, y), targets=(u, v))
```

> [!TIP]
> Source variables are optional if they are literals, in which case they are automatically added.
> The variables `x` and `y` in the previous example could have been omitted.
>
> ```python
> # Equivalent ways to define the previous functions
> f = Function(u)
> g = Function((u, v))
> ```

## Lazy evaluation

If you want to re-evaluate a function with different values, you can do so by setting the values of the source variables and then calling the `evaluate` method.

```python
# Lazy re-evaluation with different values
x.set(3)
y.set(4)
g.evaluate()
print("u =", u())  # u = 12
print("v =", v())  # v = 7
```

## Forward-mode differentiation

> In forward mode, the derivatives assigned to the source variables are propagated through the program **in the order of evaluation**.
> Use forward mode when the number of source variables is smaller than or equal to the number of target variables.

A typical use case is to compute the tangent vector to a curve $\gamma \colon \mathbb{R} \to \mathbb{R}^n$.

```python
# Tangent vector to a circle (forward-mode differentiation)
t = var(0)
x, y = var(cos(t)), var(sin(t))

γ = Function((x, y))
γ.push_tangent_at(t)    # compute tangent at (x,y)=(1,0)
print("dx/dt =", d(x))  # dx/dt = 0.0
print("dy/dt =", d(y))  # dy/dt = 1.0
```

The tangent vector in the above example is a special case of the Jacobian matrix.
In general, if your program computes a function $f \colon \mathbb{R}^m \to \mathbb{R}^n$, $x \mapsto y$, then calling `push_tangent_at(x)` computes the Jacobian matrix

$$
J_f(x) = \begin{bmatrix}
    \frac{\partial f_1}{\partial x_1} \ \ldots\ \frac{\partial f_1}{\partial x_m} \\
    \vdots \\
    \frac{\partial f_n}{\partial x_1} \ \ldots\ \frac{\partial f_n}{\partial x_m}
\end{bmatrix} \in \mathbb{R}^{n \times m}
$$

and stores it in `d(y)`.

The `push_tangent_at(seed: Variable)` method is really a convenience function for the more general `push_tangent` method and performs the following steps:

1. Set the derivative of the source variable `seed` to the identity map
2. Set the derivative of any other source variable to zero (with appropriate dimensions)
3. Call the `push_tangent` method to compute intermediate and output derivatives.

```python
# Equivalent to the previous example
t.set_derivative(1)     # scalar identity
γ.push_tangent()
print("dx/dt =", d(x))  # dx/dt = 0.0
print("dy/dt =", d(y))  # dy/dt = 1.0
```

By manually setting the derivatives of all (!) source variables, you can compute any Jacobian-vector product $\delta y = J_f(x) \cdot \delta x$ (or Jacobian-derivative product) without actually computing the Jacobian matrix.

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

## Reverse-mode differentiation (aka backpropagation)

> In reverse mode, the derivatives assigned to the target variables are propagated through the program **in the reverse order of evaluation**.
> Use reverse mode when the number of target variables is smaller than the number of source variables.

A typical use case is to compute the gradient of a scalar function $f \colon \mathbb{R}^m \to \mathbb{R}$.

```python
# Gradient of the vector norm (reverse-mode differentiation)
x = var(np.array([1, 2, 3]))
y = var(norm(x))       # L²-norm

f = Function(y)        # f : R³ → R, x ↦ y = ||x||
f.pull_gradient_at(y)
print("∇f =", d(x))    # ∇f = [[0.26726124 0.53452248 0.80178373]]
```

Note that the gradient of a scalar function is a $1 \times m$ Jacobian matrix (aka "row vector").
Analogously to [forward mode](#forward-mode-differentiation), if your program computes a function $f \colon \mathbb{R}^m \to \mathbb{R}^n$, $x \mapsto y$, then calling `pull_gradient_at(y)` computes the Jacobian matrix $J_f(x) \in \mathbb{R}^{n \times m}$ and stores it in `d(x)`.

> [!IMPORTANT]
> Given a function $f \colon M \to \mathbb{R}$ and a point $p \in M$, the [gradient](https://en.wikipedia.org/wiki/Gradient) $\nabla f(p)$ is usually defined as the unique vector such that $\langle \nabla f(p), v \rangle = {\rm d}f_p(v)$ for all tangent vectors $v \in T_pM$.
>
> 1. This definition of the gradient is **non-canonical** because it requires an extra inner product $\langle \cdot,\cdot \rangle$ on the tangent space $T_pM$.
> 2. Vectors are pushed forward by the derivative, while covectors are pulled back.
> A "gradient vector" cannot be pulled back using backpropagation (without an inner product).
>
> The term "gradient" appears in the AutoDiff API due to its frequent use in automatic differentiation. However, for mathematical consistency, we use "gradient" to refer to the differential ${\rm d}f_p$ which is a covector (represented by a "row vector"), not a vector ("column vector").

And again, the `pull_gradient_at(seed: Variable)` method is a convenience function performing the following steps:

1. Set the derivative of the target variable `seed` to the identity map
2. Set the derivative of any other target variable to zero (with appropriate dimensions)
3. Call the `pull_gradient` method to compute intermediate and input gradients

```python
# Equivalent to the previous example
y.set_derivative(1)  # scalar identity
f.pull_gradient()
print("∇f =", d(x))  # ∇f = [[0.26726124 0.53452248 0.80178373]]
```

## Advanced: changing the program after evaluation

During the first lazy evaluation or differentiation, the function is being *compiled* if necessary.
Compilation builds an internal representation of the program that permits efficient evaluation and differentiation.

```python
# Automatic compilation on first differentiation
x = var(1)
y = var(2)
u = var(x * y)

f = Function(u)
print(f.compiled())    # False
f.pull_gradient_at(u)
print(f.compiled())    # True
```

After compiling the function, you can still modify its program by assigning a new expression to one of its variables.
However, if you change the expression of a non-source variable, you must then explicitly recompile the function using the `compile` method.
Otherwise, the program might crash or produce incorrect results.

```python
# ...continuing from the previous example
a = var(3)
u.set(a**2)        # change an expression after compilation
f.compile()        # MUST recompile the program
f.evaluate()
print("u =", u())  # u = 9
```

You can also call the `compile` method before the first evaluation or differentiation to avoid the (small) overhead of compiling the program then.

```python
f = Function(u)
f.compile()
print(f.compiled())    # True
f.pull_gradient_at(u)  # no compilation needed
```
