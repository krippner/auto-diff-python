// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "common.hpp"

#include <AutoDiff/Basic>
#include <AutoDiff/Python/ExpressionBinding.hpp>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(MODULE_NAME, module)
{
    module.attr("__version__") = VERSION_INFO;
    // the module docstring is added directly to `src-python/autodiff/scalar.py`

    defCore(module); // must be called before ExpressionBinding

    using Binding = AutoDiff::Python::ExpressionBinding<double, double>;
    auto binding  = Binding(module, "Scalar");

    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(binding, "add", operator+, "")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(binding, "sub", operator-, "")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(binding, "mul", operator*, "")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(binding, "truediv", operator/, "")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(binding, "pow", pow, "")

    AUTODIFF_PYTHON_DEF_METHOD(binding, "neg", operator-, "")

    AUTODIFF_PYTHON_DEF_UNARY_OP(Binding, module, "cos", cos, "")
    AUTODIFF_PYTHON_DEF_UNARY_OP(Binding, module, "exp", exp, "")
    AUTODIFF_PYTHON_DEF_UNARY_OP(Binding, module, "log", log, "")
    AUTODIFF_PYTHON_DEF_UNARY_OP(Binding, module, "maximum", max, "")
    AUTODIFF_PYTHON_DEF_UNARY_OP(Binding, module, "minimum", min, "")
    AUTODIFF_PYTHON_DEF_UNARY_OP(Binding, module, "sin", sin, "")
    AUTODIFF_PYTHON_DEF_UNARY_OP(Binding, module, "sqrt", sqrt, "")
    AUTODIFF_PYTHON_DEF_UNARY_OP(Binding, module, "square", square, "")
}
