// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_PYTHON_EXPRESSION_BINDING_HPP
#define AUTODIFF_PYTHON_EXPRESSION_BINDING_HPP

#include "Expression.hpp"
#include "Operation.hpp"
#include "Variable.hpp"

#include <AutoDiff/src/Core/AbstractVariable.hpp>
#include <AutoDiff/src/Core/Variable.hpp> // var
#include <pybind11/pybind11.h>

#include <string>
#include <utility> // move

namespace AutoDiff::Python {

template <typename Value_, typename Derivative>
struct ExpressionBinding {
    using Value     = Value_;
    using Expr      = Python::Expression<Value, Derivative>;
    using Op        = Python::Operation<Value, Derivative>;
    using Var       = Python::Variable<Value, Derivative>;
    using ExprClass = pybind11::class_<Expr>;
    using OpClass   = pybind11::class_<Op, Expr>;
    using VarClass  = pybind11::class_<Var, Expr, AbstractVariable>;

    using Scalar     = double;
    using ScalarExpr = Python::Expression<Scalar, Derivative>;

    ExpressionBinding(pybind11::module& module, std::string const& name)
        : mExprClass{module, (name + "Expression").c_str()}
        , mOpClass{module, (name + "Operation").c_str()}
        , mVarClass{module, (name + "Variable").c_str()}
    {
        mExprClass.doc()
            = R"doc(Composition of literals, variables, and other expressions.

Expression objects are the results of operators and functions calls in AutoDiff.

Examples
--------
>>> x = var(..)  # x is a variable, an expression subclass

>>> u = x + 1    # u is an operation, an expression subclass

>>> exp(x)       # `exp` accepts both variables...

>>> exp(u)       # ...and operations.)doc";

        mOpClass.doc() = R"doc(Holds instructions for evaluating an expression.

Unlike variables, operations do not store a value or derivative.
Use the `var` function to evaluate an operation to a variable and
access its value and derivative.

Examples
--------
>>> x = var(..)  # some variable

>>> u = x + 1    # operation that represents x + 1

>>> a = var(u)   # variable that evaluates x + 1)doc";

        mVarClass.doc()
            = R"doc(Evaluates an expression and caches its value and derivative.

The expression can be a literal or a composition of operations.
An AutoDiff Variable behaves similar to a mathematical variable
in the sense that it is essentially a label pointing to a shared resource.

Variables can make computations more efficient because they allow to evaluate
an expression once and then reuse the cached result in other expressions.
Iterative computations require variables to accumulate expressions.

Examples
--------
>>> x = var(..)           # create a variable

>>> x.set(..)             # set its value or expression

>>> x.set_derivative(..)  # set its derivative

>>> x()                   # get its value

>>> d(x)                  # get its derivative
)doc";

        mVarClass.def(pybind11::init<Value>(), pybind11::arg("value") = Value{},
            R"doc(Create a variable holding a literal.)doc");

        mVarClass.def("__call__", &Var::value,
            pybind11::return_value_policy::reference_internal,
            R"doc(Returns the cached value.)doc");

        mVarClass.def(
            "set",
            [](Var const& variable, Value value) {
                variable.set(std::move(value));
            },
            pybind11::arg("value"),
            R"doc(Assign a literal to replace the current value or expression.)doc");

        mVarClass.def(
            "set",
            [](Var const& variable, Expr const& expression) {
                variable.set(expression);
            },
            pybind11::arg("expression"),
            R"doc(Assign an expression to replace the current value or expression.

The new expression is immediately evaluated (eager evaluation).)doc");

        mVarClass.def("set_derivative", &Var::setDerivative,
            pybind11::arg("derivative"),
            R"doc(Set the derivative.

The derivative is propagated during forward- or reverse-mode
automatic differentiation with a `Function` object.)doc");

        module.def(
            "var", [](Value value) { return Var{var(std::move(value))}; },
            pybind11::arg("value") = Value{},
            R"doc(Create a variable holding a literal.

The value is stored in the variable and can be accessed with the `()` method.)doc");

        module.def(
            "var",
            [](Expr const& expression) {
                return Var{var(expression.wrapper())};
            },
            pybind11::arg("expression"),
            R"doc(Create a variable that evaluates an expression of other variables.

The expression is immediately evaluated (eager evaluation).)doc");

        module.def(
            "d", [](Var const& variable) { return variable.derivative(); },
            pybind11::return_value_policy::reference_internal,
            pybind11::arg("variable"),
            R"doc(Returns the differential (i.e., the cached derivative) of a variable.

Depending on the mode of differentiation, this derivative
can be a tangent vector or gradient.)doc");
    }

    // A @ B, A @ BLiteral
    template <typename FuncExpr, typename FuncValue>
    void defInfixOp(std::string const& name, FuncExpr&& funcExpr,
        FuncValue&& funcValue, std::string const& description)
    {
        mExprClass.def(("__" + name + "__").c_str(), funcExpr,
            pybind11::arg("other"), description.c_str());
        mExprClass.def(("__" + name + "__").c_str(), funcValue,
            pybind11::arg("other"), description.c_str());
    }

    // BLiteral @ A
    template <typename FuncRValue>
    void defRInfixOp(std::string const& name, FuncRValue&& funcRValue,
        std::string const& description)
    {
        mExprClass.def(("__r" + name + "__").c_str(), funcRValue,
            pybind11::arg("other"), description.c_str());
    }

    // A @ Scalar, A @ ScalarLiteral
    template <typename FuncScalar, typename FuncScalarExpr>
    void defBroadcastInfixOp(std::string const& name, FuncScalar&& funcScalar,
        FuncScalarExpr&& funcScalarExpr, std::string const& description)
    {
        mExprClass.def(("__" + name + "__").c_str(), funcScalar,
            pybind11::arg("scalar"), description.c_str());
        mExprClass.def(("__" + name + "__").c_str(), funcScalarExpr,
            pybind11::arg("expression"), description.c_str());
    }

    // Scalar @ A, ScalarLiteral @ A
    template <typename FuncRScalar, typename FuncRScalarExpr>
    void defRBroadcastInfixOp(std::string const& name,
        FuncRScalar&& funcRScalar, FuncRScalarExpr&& funcRScalarExpr,
        std::string const& description)
    {
        mExprClass.def(("__r" + name + "__").c_str(), funcRScalar,
            pybind11::arg("scalar"), description.c_str());
        mExprClass.def(("__r" + name + "__").c_str(), funcRScalarExpr,
            pybind11::arg("expression"), description.c_str());
    }

    // e.g. negation
    template <typename Func>
    void defUnaryOp(
        std::string const& name, Func&& func, std::string const& description)
    {
        mExprClass.def(("__" + name + "__").c_str(), func, description.c_str());
    }

private:
    ExprClass mExprClass;
    OpClass mOpClass;
    VarClass mVarClass;
};

template <typename Func>
void defUnaryOp(pybind11::module& module, std::string const& name, Func&& func,
    std::string const& description)
{
    module.def(
        name.c_str(), func, pybind11::arg("operand"), description.c_str());
}

template <typename FuncExpr, typename FuncValue, typename FuncRValue>
void defBinaryOp(pybind11::module& module, std::string const& name,
    FuncExpr&& funcExpr, FuncValue&& funcValue, FuncRValue&& funcRValue,
    std::string const& description)
{
    module.def(name.c_str(), funcExpr, pybind11::arg("lhs"),
        pybind11::arg("rhs"), description.c_str());
    module.def(name.c_str(), funcValue, pybind11::arg("lhs"),
        pybind11::arg("rhs"), description.c_str());
    module.def(name.c_str(), funcRValue, pybind11::arg("lhs"),
        pybind11::arg("rhs"), description.c_str());
}

} // namespace AutoDiff::Python

#define AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(                                      \
    binding, name, operation, description)                                     \
    {                                                                          \
        using Binding = decltype(binding);                                     \
                                                                               \
        auto funcExpr = [](Binding::Expr const& x, Binding::Expr const& y) {   \
            return Binding::Op{operation(x.wrapper(), y.wrapper())};           \
        };                                                                     \
        auto funcValue = [](Binding::Expr const& x, Binding::Value y) {        \
            return Binding::Op{operation(x.wrapper(), std::move(y))};          \
        };                                                                     \
        auto funcRValue = [](Binding::Expr const& y, Binding::Value x) {       \
            return Binding::Op{operation(std::move(x), y.wrapper())};          \
        };                                                                     \
        binding.defInfixOp(name, funcExpr, funcValue, description);            \
        binding.defRInfixOp(name, funcRValue, description);                    \
    }

#define AUTODIFF_PYTHON_DEF_INFIX_OP(                                          \
    bindingX, bindingY, Binding, name, operation, description)                 \
    {                                                                          \
        using BindingX = decltype(bindingX);                                   \
        using BindingY = decltype(bindingY);                                   \
                                                                               \
        auto funcExpr = [](BindingX::Expr const& x, BindingY::Expr const& y) { \
            return Binding::Op{operation(x.wrapper(), y.wrapper())};           \
        };                                                                     \
        auto funcValue = [](BindingX::Expr const& x, BindingY::Value y) {      \
            return Binding::Op{operation(x.wrapper(), std::move(y))};          \
        };                                                                     \
        bindingX.defInfixOp(name, funcExpr, funcValue, description);           \
        if constexpr (std::is_arithmetic_v<BindingX::Value>) {                 \
            auto funcRValue = [](BindingY::Expr const& y, BindingX::Value x) { \
                return Binding::Op{operation(std::move(x), y.wrapper())};      \
            };                                                                 \
            bindingY.defRInfixOp(name, funcRValue, description);               \
        }                                                                      \
    }

#define AUTODIFF_PYTHON_DEF_METHOD(binding, name, operation, description)      \
    {                                                                          \
        using Binding = decltype(binding);                                     \
                                                                               \
        auto func = [](Binding::Expr const& x) {                               \
            return Binding::Op{operation(x.wrapper())};                        \
        };                                                                     \
        binding.defUnaryOp(name, func, description);                           \
    }

#define AUTODIFF_PYTHON_DEF_UNARY_OP(                                          \
    Binding, module, name, operation, description)                             \
    {                                                                          \
        auto func = [](Binding::Expr const& x) {                               \
            return Binding::Op{operation(x.wrapper())};                        \
        };                                                                     \
        AutoDiff::Python::defUnaryOp(module, name, func, description);         \
    }

#define AUTODIFF_PYTHON_DEF_REDUCTION(                                         \
    BindingX, Binding, module, name, operation, description)                   \
    {                                                                          \
        auto func = [](BindingX::Expr const& x) {                              \
            return Binding::Op{operation(x.wrapper())};                        \
        };                                                                     \
        AutoDiff::Python::defUnaryOp(module, name, func, description);         \
    }

#define AUTODIFF_PYTHON_DEF_BINARY_OP(                                         \
    BindingX, BindingY, Binding, module, name, operation, description)         \
    {                                                                          \
        auto funcExpr = [](BindingX::Expr const& x, BindingY::Expr const& y) { \
            return Binding::Op{operation(x.wrapper(), y.wrapper())};           \
        };                                                                     \
        auto funcValue = [](BindingX::Expr const& x, BindingY::Value y) {      \
            return Binding::Op{operation(x.wrapper(), std::move(y))};          \
        };                                                                     \
        auto funcRValue = [](BindingX::Value x, BindingY::Expr const& y) {     \
            return Binding::Op{operation(std::move(x), y.wrapper())};          \
        };                                                                     \
        AutoDiff::Python::defBinaryOp(                                         \
            module, name, funcExpr, funcValue, funcRValue, description);       \
    }

#define AUTODIFF_PYTHON_DEF_BROADCAST_INFIX_OP(                                \
    binding, name, operation, description)                                     \
    {                                                                          \
        using Binding = decltype(binding);                                     \
                                                                               \
        auto funcScalar = [](Binding::Expr const& x, Binding::Scalar y) {      \
            return Binding::Op{operation(x.wrapper(), std::move(y))};          \
        };                                                                     \
        auto funcScalarExpr                                                    \
            = [](Binding::Expr const& x, Binding::ScalarExpr const& y) {       \
                  return Binding::Op{operation(x.wrapper(), y.wrapper())};     \
              };                                                               \
        binding.defBroadcastInfixOp(                                           \
            name, funcScalar, funcScalarExpr, description);                    \
    }

#define AUTODIFF_PYTHON_DEF_R_BROADCAST_INFIX_OP(                              \
    binding, name, operation, description)                                     \
    {                                                                          \
        using Binding = decltype(binding);                                     \
                                                                               \
        auto funcRScalar = [](Binding::Expr const& y, Binding::Scalar x) {     \
            return Binding::Op{operation(std::move(x), y.wrapper())};          \
        };                                                                     \
        auto funcRScalarExpr                                                   \
            = [](Binding::Expr const& y, Binding::ScalarExpr const& x) {       \
                  return Binding::Op{operation(x.wrapper(), y.wrapper())};     \
              };                                                               \
        binding.defRBroadcastInfixOp(                                          \
            name, funcRScalar, funcRScalarExpr, description);                  \
    }

#endif // AUTODIFF_PYTHON_EXPRESSION_BINDING_HPP
