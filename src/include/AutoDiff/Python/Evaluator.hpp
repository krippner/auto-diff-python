// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_PYTHON_EVALUATOR_HPP
#define AUTODIFF_PYTHON_EVALUATOR_HPP

#include <AutoDiff/src/Core/AbstractVariable.hpp>
#include <AutoDiff/src/Core/Expression.hpp> // ValueType
#include <AutoDiff/src/internal/traits.hpp> // Evaluated

#include <memory>
#include <type_traits> // enable_if_t
#include <utility>     // move

namespace AutoDiff::Python {

template <typename Value_, typename Derivative_>
class AbstractEvaluator {
public:
    using Value      = Value_;
    using Derivative = Derivative_;

    virtual ~AbstractEvaluator() = default;

    virtual void transferChildrenTo(internal::Node& node) = 0;
    virtual auto value() -> Value const&                  = 0;
    virtual auto pushForward() -> Derivative const&       = 0;
    virtual void pullBack(Derivative const& gradient)     = 0;
    virtual void releaseCache()                           = 0;
};

template <typename Expr>
struct EvaluatorType {
    using Value      = internal::Evaluated_t<ValueType_t<Expr>>;
    using Derivative = typename Expr::Derivative;
    using type       = AbstractEvaluator<Value, Derivative>;
};

template <typename Expr>
using EvaluatorType_t = typename EvaluatorType<Expr>::type;

template <typename Expr, typename = void>
class Evaluator : public EvaluatorType_t<Expr> {
public:
    using Base = EvaluatorType_t<Expr>;
    using typename Base::Derivative;
    using typename Base::Value;

    explicit Evaluator(Expr expression)
        : mExpression{std::move(expression)}
    {
    }

    void transferChildrenTo(internal::Node& node) final
    {
        mExpression._transferChildrenTo(node);
    }

    [[nodiscard]] auto value() -> Value const& final
    {
        mValuePtr = std::make_unique<Value>(mExpression._value());
        return *mValuePtr;
    }

    [[nodiscard]] auto pushForward() -> Derivative const& final
    {
        mDerivativePtr
            = std::make_unique<Derivative>(mExpression._pushForward());
        return *mDerivativePtr;
    }

    void pullBack(Derivative const& gradient) final
    {
        mExpression._pullBack(gradient);
    }

    void releaseCache() final
    {
        mValuePtr.reset();
        mDerivativePtr.reset();
        mExpression._releaseCache();
    }

private:
    Expr mExpression;

    // cache
    std::unique_ptr<Value> mValuePtr;
    std::unique_ptr<Derivative> mDerivativePtr;
};

template <typename Var>
class Evaluator<Var, std::enable_if_t<std::is_base_of_v<AbstractVariable, Var>>>
    : public EvaluatorType_t<Var> {
public:
    using Base = EvaluatorType_t<Var>;
    using typename Base::Derivative;
    using typename Base::Value;

    explicit Evaluator(Var variable)
        : mVariable{std::move(variable)}
    {
    }

    void transferChildrenTo(internal::Node& node) final
    {
        mVariable._transferChildrenTo(node);
    }

    [[nodiscard]] auto value() -> Value const& final
    {
        return mVariable._value();
    }

    [[nodiscard]] auto pushForward() -> Derivative const& final
    {
        return mVariable._pushForward();
    }

    void pullBack(Derivative const& gradient) final
    {
        mVariable._pullBack(gradient);
    }

    void releaseCache() final { } // no cache

private:
    Var mVariable;
};

} // namespace AutoDiff::Python

#endif // AUTODIFF_PYTHON_EVALUATOR_HPP
