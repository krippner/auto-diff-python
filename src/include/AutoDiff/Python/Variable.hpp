// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_PYTHON_VARIABLE_HPP
#define AUTODIFF_PYTHON_VARIABLE_HPP

#include "Expression.hpp"

#include <AutoDiff/src/Core/Variable.hpp> // Variable, d

#include <utility> // move

namespace AutoDiff::Python {

template <typename Value, typename Derivative>
class Variable : public AbstractVariable, public Expression<Value, Derivative> {
public:
    explicit Variable(Value value)
        : mVariable{std::move(value)}
    {
    }

    explicit Variable(AutoDiff::Variable<Value, Derivative> variable)
        : mVariable{std::move(variable)}
    {
    }

    ~Variable() override = default;

    Variable(Variable const&)                        = default;
    Variable(Variable&&) noexcept                    = default;
    auto operator=(Variable const&) -> Variable&     = default;
    auto operator=(Variable&&) noexcept -> Variable& = default;

    [[nodiscard]] auto value() const -> Value const& { return mVariable(); }

    void set(Value value) const { mVariable = std::move(value); }

    void set(Expression<Value, Derivative> const& expression) const
    {
        mVariable.setExpression(expression.wrapper());
    }

    [[nodiscard]] auto derivative() const -> Derivative const&
    {
        return d(mVariable);
    }

    void setDerivative(Derivative derivative) const
    {
        mVariable.setDerivative(std::move(derivative));
    }

    [[nodiscard]] auto _node() const -> internal::AbstractComputation* override
    {
        return mVariable._node();
    }

    [[nodiscard]] auto
    wrapper() const -> ExpressionWrapper<Value, Derivative> override
    {
        return ExpressionWrapper<Value, Derivative>(mVariable);
    }

private:
    AutoDiff::Variable<Value, Derivative> mVariable;
};

} // namespace AutoDiff::Python

#endif // AUTODIFF_PYTHON_VARIABLE_HPP
