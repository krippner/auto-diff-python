# Variables and expressions

## Variables

A *variable* represents shared persistent state in your program.
Variables store input values, intermediate and output results, and cache derivatives.
They are created by passing literals (e.g. scalars and arrays) or expressions to the `var` function.

```python
# Creating variables
x = var(42)           # create an input variable storing a literal
y = var(x + 2)        # create a variable evaluating an expression

# Getters
x()                   # get the value of x
d(x)                  # get the derivative of x

# Setters
x.set(3.14)           # set the value of x
x.set_derivative(1.0) # set the derivative of x
y.set(x**2 - 1)       # set a new expression for y
```

Variables are evaluated eagerly, meaning that their value is computed immediately upon creation or update.
This is the most intuitive behavior, just like with regular Python variables, and is especially useful for debugging and testing, because you can inspect the value of an expression and locate errors more easily.

```python
# Eager evaluation
x = var(1)
print("x =", x())  # x = 1
y = var(x + 2)
print("y =", y())  # y = 3
y.set(2 * x)
print("y =", y())  # y = 2
```

## Expressions

Combining variables and literals with operators and function calls creates *expressions*.
They generally do not cache any value or derivative, and only keep track of the computation to be performed when the expression is finally evaluated or differentiated.

```python
# Examples of expressions
x = var(3.14)      # a variable itself is also an expression
u = x + 2          # operation combining a variable and a literal
v = sin(u)         # function applied to an expression
```

## Variables vs. expressions

During gradient computation, derivatives are accumulated in reverse order of evaluation.
To speed up computation, the results from the evaluation pass need be stored in memory.
Doing so for every operation, however, could lead to excessive memory usage.

Expressions help mitigate this issue by not storing any value, significantly reducing the memory footprint.
As a rule of thumb, convert an expression to a variable (using `var`) if

- it is used in multiple expressions,

    ```python
    x = var(..)
    u, v = x + 2, x * 3 # evaluates x only once
    ```

- it is updated in a loop,

    ```python
    x = var(..)
    for i in range(10):
        x.set(x + 1)   # BAD: cyclic dependency, will raise an error later
        x = x + 1      # BAD: not what you want
        x = var(x + 1) # GOOD: evaluate to a NEW variable
    ```

- or you need to access its value or derivative.

Otherwise, use benchmarks to see whether introducing a variable leads to a significant speedup.
Giving you control over this space-time trade-off is a key aspect of the API design.
