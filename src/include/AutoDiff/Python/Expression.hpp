// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_PYTHON_EXPRESSION_HPP
#define AUTODIFF_PYTHON_EXPRESSION_HPP

#include "Evaluator.hpp"

#include <AutoDiff/src/Core/Expression.hpp>

#include <memory>

namespace AutoDiff::Python {

// dynamic polymorphism inside complete type
template <typename Value, typename Derivative_>
class ExpressionWrapper : public AutoDiff::Expression<
                              Python::ExpressionWrapper<Value, Derivative_>> {
public:
    using Derivative = Derivative_;

    template <typename Expr>
    explicit ExpressionWrapper(Expr expression)
        : mEvaluator{std::make_shared<Evaluator<Expr>>(std::move(expression))}
    {
    }

    explicit ExpressionWrapper(
        std::shared_ptr<AbstractEvaluator<Value, Derivative>> evaluator)
        : mEvaluator{std::move(evaluator)}
    {
    }

    ~ExpressionWrapper() = default;

    ExpressionWrapper(ExpressionWrapper const&)                    = default;
    ExpressionWrapper(ExpressionWrapper&&) noexcept                = default;
    auto operator=(ExpressionWrapper const&) -> ExpressionWrapper& = default;
    auto operator=(
        ExpressionWrapper&&) noexcept -> ExpressionWrapper& = default;

    [[nodiscard]] auto _valueImpl() -> Value const&
    {
        return mEvaluator->value();
    }

    [[nodiscard]] auto _pushForwardImpl() -> Derivative const&
    {
        return mEvaluator->pushForward();
    }

    void _pullBackImpl(Derivative const& derivative)
    {
        mEvaluator->pullBack(derivative);
    }

    void _transferChildrenToImpl(internal::Node& node) const
    {
        mEvaluator->transferChildrenTo(node);
    }

    void _releaseCacheImpl() const { mEvaluator->releaseCache(); }

private:
    // shared to allow copy
    std::shared_ptr<AbstractEvaluator<Value, Derivative>> mEvaluator;
};

template <typename Value, typename Derivative>
class Expression {
public:
    [[nodiscard]] virtual auto
    wrapper() const -> ExpressionWrapper<Value, Derivative> = 0;
};

} // namespace AutoDiff::Python

#endif // AUTODIFF_PYTHON_EXPRESSION_HPP
