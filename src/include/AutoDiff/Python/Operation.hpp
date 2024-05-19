// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_PYTHON_OPERATION_HPP
#define AUTODIFF_PYTHON_OPERATION_HPP

#include "Evaluator.hpp"
#include "Expression.hpp"

#include <memory>

namespace AutoDiff::Python {

template <typename Value, typename Derivative>
class Operation : public Expression<Value, Derivative> {
public:
    template <typename Op>
    explicit Operation(Op operation)
        : mEvaluator{std::make_shared<Evaluator<Op>>(std::move(operation))}
    {
    }

    ~Operation() = default;

    Operation(Operation const&)                        = default;
    Operation(Operation&&) noexcept                    = default;
    auto operator=(Operation const&) -> Operation&     = default;
    auto operator=(Operation&&) noexcept -> Operation& = default;

    [[nodiscard]] auto
    wrapper() const -> ExpressionWrapper<Value, Derivative> override
    {
        return ExpressionWrapper<Value, Derivative>(mEvaluator);
    }

private:
    std::shared_ptr<AbstractEvaluator<Value, Derivative>> mEvaluator;
};

} // namespace AutoDiff::Python

#endif // AUTODIFF_PYTHON_OPERATION_HPP
