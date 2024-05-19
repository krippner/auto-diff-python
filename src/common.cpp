// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "common.hpp"

#include <AutoDiff/src/Core/AbstractVariable.hpp>
#include <AutoDiff/src/Core/Function.hpp>

namespace py = pybind11;

using AutoDiff::AbstractVariable;
using AutoDiff::Function;

namespace detail {

auto createSources(py::tuple const& sourcesTuple) -> Function::Sources
{
    auto sources = Function::Sources{};
    for (auto const& source : sourcesTuple) {
        sources.obj.insert(source.cast<AbstractVariable const&>()._node());
    }
    return sources;
}

auto createTargets(py::tuple const& targetsTuple) -> Function::Targets
{
    auto targets = Function::Targets{};
    for (auto const& target : targetsTuple) {
        targets.obj.insert(target.cast<AbstractVariable const&>()._node());
    }
    return targets;
}

} // namespace detail

void defCore(py::module& module)
{
    py::class_<AbstractVariable>(module, "Variable", py::module_local(),
        "Base class for all variables.");

    auto function
        = py::class_<Function>(module, "Function", py::module_local());

    function.doc()
        = R"doc(Represents a program defined by target variables as functions
of source variables for evaluation and differentiation.

In maths, the space containing sources or targets is usually called the
function domain or codomain, respectively.

Note 1
------
Generally, the function needs to be evaluated before differentiating,
either lazily during expression construction or explicitly by calling
`evaluate`.

Note 2
------
After assigning a new expression to one of the variables involved,
the function must be re-compiled by calling the `compile` method.
This is necessary because a `Function` object is just a view into the internal
computation graph and it holds only non-owning references to the computation
nodes (which are owned by variables).)doc";

    function.def(
        py::init<>([](py::tuple const& targets, py::tuple const& sources) {
            return Function(
                detail::createSources(sources), detail::createTargets(targets));
        }),
        py::arg("targets"), py::kw_only(), py::arg("sources") = py::tuple(),
        R"doc(Create a function mapping sources to targets.

The source variables are used to limit the search for dependencies.
This can be useful to partition the computation graph into subgraphs.

Parameters
----------
targets : tuple of Variable
          The target variables. Must not be empty.
sources : tuple of Variable, optional
          The source variables.
          Need not be the actual sources of the function.
          
Examples
--------
>>> x = var(..)  # literal variable

>>> u, v = expression_1(x)

>>> a, b = expression_2(u, v)

>>> f_1_2 = Function((a, b))                        # x ↦ (a, b)

>>> f_2 = Function(sources=(u, v), targets=(a, b))  # (u, v) ↦ (a, b)

Raises
------
RuntimeError
    If the function has no targets.)doc");

    function.def(py::init<>([](AbstractVariable const& target,
                                py::tuple const& sources) {
        return Function(detail::createSources(sources),
            detail::createTargets(py::make_tuple(target)));
    }),
        py::arg("target"), py::kw_only(), py::arg("sources") = py::tuple(),
        R"doc(Create a function mapping sources to a single target.

The source variables are used to limit the search for dependencies.
This can be useful to partition the computation graph into subgraphs.

Parameters
----------
target : Variable
         The target variable.
sources : tuple of Variable, optional
          The source variables.
          Need not be the actual sources of the function.
          
Examples
--------
>>> x = var(..)  # literal variable

>>> u, v = expression_1(x)

>>> a = expression_2(u, v)

>>> f_1_2 = Function(a)                       # x ↦ a

>>> f_2 = Function(sources=(u, v), target=a)  # (u, v) ↦ a)doc");

    function.def(py::init<>([](py::tuple const& targets,
                                AbstractVariable const& source) {
        return Function(detail::createSources(py::make_tuple(source)),
            detail::createTargets(targets));
    }),
        py::arg("targets"), py::kw_only(), py::arg("source"),
        R"doc(Create a function mapping sources to targets.

The source variable is used to limit the search for dependencies.
This can be useful to partition the computation graph into subgraphs.

Parameters
----------
targets : tuple of Variable
          The target variables. Must not be empty.
source : Variable
         A source variable.
         Need not be an actual source of the function.

Examples
--------
>>> x = var(..)  # literal variable

>>> u = expression_1(x)

>>> a, b = expression_2(u)

>>> f_1_2 = Function(source=x, targets=(a, b))  # x ↦ (a, b)

>>> f_2 = Function(source=u, targets=(a, b))    # u ↦ (a, b)

Raises
------
RuntimeError
    If the function has no targets.)doc");

    function.def(py::init<>([](AbstractVariable const& target,
                                AbstractVariable const& source) {
        return Function(detail::createSources(py::make_tuple(source)),
            detail::createTargets(py::make_tuple(target)));
    }),
        py::arg("target"), py::kw_only(), py::arg("source"),
        R"doc(Create a function mapping sources to a single target.

The source variable is used to limit the search for dependencies.
This can be useful to partition the computation graph into subgraphs.

Parameters
----------
target : Variable
         The target variable.
source : Variable
         The source variable.
         Need not be an actual source of the function.
         
Examples
--------
>>> x = var(..)  # literal variable

>>> u = expression_1(x)

>>> a = expression_2(u)

>>> f_1_2 = Function(source=x, target=a)  # x ↦ a

>>> f_2 = Function(source=u, target=a)    # u ↦ a)doc");

    function.def("compile", &Function::compile,
        R"doc(Compile the function for evaluation and differentiation.

Compilation generates a topologically ordered sequence of computation
references, which is used to efficiently traverse the computation graph.
It is triggered automatically before the first evaluation or differentiation.

Note
----
This method must be called after assigning a new expression to one of the
variables involved.

Raises
------
RuntimeError
    If the corresponding program has cyclic dependencies.)doc");

    function.def("compiled", &Function::compiled,
        R"doc(Returns whether the function has been compiled successfully.)doc");

    function.def("__str__", &Function::str, "For debugging purposes.");

    function.def("evaluate", &Function::evaluate,
        R"doc(Evaluate the target and intermediate variables.

Before the first evaluation, the function is automatically compiled if
necessary.

Note
----
Before calling this, all source variables must have valid values.

Raises
------
RuntimeError
    If the corresponding program has cyclic dependencies.)doc");

    function.def("push_tangent", &Function::pushTangent,
        R"doc(Forward-mode automatic differentiation.

Computes the tangent vectors at target and intermediate variables
by propagating the derivatives related to the source variables forward
along the function, i.e., in the same direction as the evaluation.

Use this method to compute the Jacobian-vector product.

Examples
--------
>>> x = var(0)            # literal variable

>>> u = var(x * 2)        # eagerly evaluated variable

>>> f = Function(u)

>>> δx = 1.0              # (scalar) tangent vector

>>> x.set_derivative(δx)  # seed forward propagation

>>> f.push_tangent()      # compute the Jacobian-vector product

>>> d(u)                  # δu = ∂u/∂x * δx = 2.0

Note
----
Before calling this, the function must be evaluated and all source
variables must have valid derivatives.

Raises
------
RuntimeError
    If the corresponding program has cyclic dependencies.)doc");

    function.def("push_tangent_at", &Function::pushTangentAt, py::arg("seed"),
        R"doc(Forward-mode automatic differentiation with seed.

Differentiates the target and intermediate variables of the function
with respect to a specified source variable (seed).

Use this method to compute the Jacobian matrix.

Parameters
----------
seed : Variable
       The source variable used to seed propagation.

Examples
--------
>>> x = var(0)            # literal variable

>>> u = var(x * 2)        # eagerly evaluated variable

>>> f = Function(u)

>>> f.push_tangent_at(x)  # compute the Jacobian matrix

>>> d(u)                  # ∂u/∂x = 2.0

Note
----
Before calling this, the function must be evaluated.

Raises
------
RuntimeError
    If the corresponding program has cyclic dependencies.
RuntimeError
    If the seed is not an actual source of the function.)doc");

    function.def("pull_gradient", &Function::pullGradient,
        R"doc(Reverse-mode automatic differentiation (backpropagation).

Computes the gradients with respect to source and intermediate variables
by propagating the derivatives related to the target variables backward
along this function, i.e., in the opposite direction of the evaluation.

Examples
--------
>>> x = var(0)             # literal variable

>>> u = var(x * 2)         # eagerly evaluated variable

>>> f = Function(u)

>>> ∇_u = 1.0              # (scalar) gradient w.r.t. u

>>> u.set_derivative(∇_u)  # seed backpropagation

>>> f.pull_gradient()

>>> d(x)                   # ∇_x = ∇_u * ∂u/∂x = 2.0

Note
----
Before calling this, the function must be evaluated and all target
variables must have valid derivatives.

Raises
------
RuntimeError
    If the corresponding program has cyclic dependencies.)doc");

    function.def("pull_gradient_at", &Function::pullGradientAt, py::arg("seed"),
        R"doc(Reverse-mode automatic differentiation (backpropagation) with seed.

Differentiates the specified target variable (seed) with respect
to the source and intermediate variables of the function.

Use this method to compute the gradient.

Parameters
----------
seed : Variable
       The target variable used to seed backpropagation.

Examples
--------
>>> x = var(0)             # literal variable

>>> u = var(x * 2)         # eagerly evaluated variable

>>> f = Function(u)

>>> f.pull_gradient_at(u)  # compute the gradient (or Jacobian matrix)

>>> d(x)                   # ∇_x = ∂u/∂x = 2.0

Note
----
Before calling this, the function must be evaluated.

Raises
------
RuntimeError
    If the corresponding program has cyclic dependencies.
RuntimeError
    If the seed is not a target of the function.)doc");
}
