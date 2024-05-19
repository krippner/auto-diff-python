// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "common.hpp"

#include <AutoDiff/Eigen>
#include <AutoDiff/Python/ExpressionBinding.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(MODULE_NAME, module)
{
    module.attr("__version__") = VERSION_INFO;
    // the module docstring is added directly to `src-python/autodiff/array.py`

    /*
     * Notes:
     *
     * 1) Expression bindings aim to resemble NumPy notation, while the C++
     * functions follow Eigen's naming conventions.
     * For example, sum (NumPy) vs. total (Eigen).
     *
     * 2) For operations taking both vectors and matrices, the vector binding
     * must be defined before the matrix binding to ensure correct overload.
     * This way, N⨉1 NumPy arrays use the vector bindings and 1⨉N arrays
     * use the matrix bindings.
     */

    defCore(module); // must be called before ExpressionBinding

    using ScalarBinding
        = AutoDiff::Python::ExpressionBinding<double, Eigen::MatrixXd>;
    auto scalarBinding = ScalarBinding(module, "Scalar");

    using VectorBinding
        = AutoDiff::Python::ExpressionBinding<Eigen::VectorXd, Eigen::MatrixXd>;
    auto vectorBinding = VectorBinding(module, "Vector");

    using MatrixBinding
        = AutoDiff::Python::ExpressionBinding<Eigen::MatrixXd, Eigen::MatrixXd>;
    auto matrixBinding = MatrixBinding(module, "Matrix");

    // scalar operations

    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(scalarBinding, "add", operator+, "")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(scalarBinding, "sub", operator-, "")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(scalarBinding, "mul", operator*, "")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(scalarBinding, "truediv", operator/, "")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(scalarBinding, "pow", pow, "")

    AUTODIFF_PYTHON_DEF_METHOD(scalarBinding, "neg", operator-, "")

    AUTODIFF_PYTHON_DEF_UNARY_OP(ScalarBinding, module, "cos", cos, "")
    AUTODIFF_PYTHON_DEF_UNARY_OP(ScalarBinding, module, "exp", exp, "")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        ScalarBinding, module, "log", log, "Natural logarithm.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        ScalarBinding, module, "maximum", max, "Maximum of a scalar and zero.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        ScalarBinding, module, "minimum", min, "Minimum of a scalar and zero.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(ScalarBinding, module, "sin", sin, "")
    AUTODIFF_PYTHON_DEF_UNARY_OP(ScalarBinding, module, "sqrt", sqrt, "")
    AUTODIFF_PYTHON_DEF_UNARY_OP(ScalarBinding, module, "square", square, "")

    // vector (cwise) operations

    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(vectorBinding, "add", operator+, "")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(vectorBinding, "sub", operator-, "")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(
        vectorBinding, "mul", cwiseProduct, "Product, element-wise.")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(
        vectorBinding, "truediv", cwiseQuotient, "Quotient, element-wise.")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(
        vectorBinding, "pow", pow, "Element-wise power of vector elements.")

    AUTODIFF_PYTHON_DEF_METHOD(vectorBinding, "neg", operator-, "")

    AUTODIFF_PYTHON_DEF_UNARY_OP(
        VectorBinding, module, "cos", cos, "Cosine, element-wise.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        VectorBinding, module, "exp", exp, "Exponential, element-wise.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        VectorBinding, module, "log", log, "Natural logarithm, element-wise.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(VectorBinding, module, "maximum", max,
        "Element-wise maximum of vector elements and zero.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(VectorBinding, module, "minimum", min,
        "Element-wise minimum of vector elements and zero.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        VectorBinding, module, "sin", sin, "Sine, element-wise.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        VectorBinding, module, "sqrt", sqrt, "Square root, element-wise.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        VectorBinding, module, "square", square, "Square, element-wise.")

    // matrix (cwise) operations

    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(matrixBinding, "add", operator+, "")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(matrixBinding, "sub", operator-, "")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(
        matrixBinding, "mul", cwiseProduct, "Product, element-wise.")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(
        matrixBinding, "truediv", cwiseQuotient, "Quotient, element-wise.")
    AUTODIFF_PYTHON_DEF_SYM_INFIX_OP(
        matrixBinding, "pow", pow, "Element-wise power of matrix elements.")

    AUTODIFF_PYTHON_DEF_METHOD(matrixBinding, "neg", operator-, "")

    AUTODIFF_PYTHON_DEF_UNARY_OP(
        MatrixBinding, module, "cos", cos, "Cosine, element-wise.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        MatrixBinding, module, "exp", exp, "Exponential, element-wise.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        MatrixBinding, module, "log", log, "Natural logarithm, element-wise.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(MatrixBinding, module, "maximum", max,
        "Element-wise maximum of matrix elements and zero.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(MatrixBinding, module, "minimum", min,
        "Element-wise minimum of matrix elements and zero.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        MatrixBinding, module, "sin", sin, "Sine, element-wise.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        MatrixBinding, module, "sqrt", sqrt, "Square root, element-wise.")
    AUTODIFF_PYTHON_DEF_UNARY_OP(
        MatrixBinding, module, "square", square, "Square, element-wise.")

    // vector (left) broadcast operations

    AUTODIFF_PYTHON_DEF_BROADCAST_INFIX_OP(vectorBinding, "add", operator+, "")
    AUTODIFF_PYTHON_DEF_BROADCAST_INFIX_OP(vectorBinding, "sub", operator-, "")
    AUTODIFF_PYTHON_DEF_BROADCAST_INFIX_OP(vectorBinding, "mul", operator*, "")
    AUTODIFF_PYTHON_DEF_BROADCAST_INFIX_OP(
        vectorBinding, "truediv", operator/, "")
    AUTODIFF_PYTHON_DEF_BROADCAST_INFIX_OP(vectorBinding, "pow", pow, "")

    // vector (right) broadcast operations

    AUTODIFF_PYTHON_DEF_R_BROADCAST_INFIX_OP(
        vectorBinding, "add", operator+, "")
    AUTODIFF_PYTHON_DEF_R_BROADCAST_INFIX_OP(
        vectorBinding, "sub", operator-, "")
    AUTODIFF_PYTHON_DEF_R_BROADCAST_INFIX_OP(
        vectorBinding, "mul", operator*, "")
    AUTODIFF_PYTHON_DEF_R_BROADCAST_INFIX_OP(
        vectorBinding, "truediv", operator/, "")

    // matrix (left) broadcast operations

    AUTODIFF_PYTHON_DEF_BROADCAST_INFIX_OP(matrixBinding, "add", operator+, "")
    AUTODIFF_PYTHON_DEF_BROADCAST_INFIX_OP(matrixBinding, "sub", operator-, "")
    AUTODIFF_PYTHON_DEF_BROADCAST_INFIX_OP(matrixBinding, "mul", operator*, "")
    AUTODIFF_PYTHON_DEF_BROADCAST_INFIX_OP(
        matrixBinding, "truediv", operator/, "")
    AUTODIFF_PYTHON_DEF_BROADCAST_INFIX_OP(matrixBinding, "pow", pow, "")

    // matrix (right) broadcast operations

    AUTODIFF_PYTHON_DEF_R_BROADCAST_INFIX_OP(
        matrixBinding, "add", operator+, "")
    AUTODIFF_PYTHON_DEF_R_BROADCAST_INFIX_OP(
        matrixBinding, "sub", operator-, "")
    AUTODIFF_PYTHON_DEF_R_BROADCAST_INFIX_OP(
        matrixBinding, "mul", operator*, "")
    AUTODIFF_PYTHON_DEF_R_BROADCAST_INFIX_OP(
        matrixBinding, "truediv", operator/, "")

    // vector-vector products

    AUTODIFF_PYTHON_DEF_BINARY_OP(VectorBinding, VectorBinding, ScalarBinding,
        module, "dot", dot, "Dot product of two vectors.")
    AUTODIFF_PYTHON_DEF_BINARY_OP(VectorBinding, VectorBinding, MatrixBinding,
        module, "outer", tensorProduct,
        "Compute the outer (tensor) product of two vectors.")

    // matrix-vector products
    // (must be defined before matrix-matrix products for correct overload)

    AUTODIFF_PYTHON_DEF_BINARY_OP(MatrixBinding, VectorBinding, VectorBinding,
        module, "matmul", operator*, "Matrix-vector product.")
    AUTODIFF_PYTHON_DEF_INFIX_OP(matrixBinding, vectorBinding, VectorBinding,
        "matmul", operator*, "Matrix-vector product.")

    // matrix-matrix products

    AUTODIFF_PYTHON_DEF_BINARY_OP(MatrixBinding, MatrixBinding, MatrixBinding,
        module, "matmul", operator*, "Matrix-matrix product.")
    AUTODIFF_PYTHON_DEF_INFIX_OP(matrixBinding, matrixBinding, MatrixBinding,
        "matmul", operator*, "Matrix-matrix product.")

    // vector reductions

    AUTODIFF_PYTHON_DEF_REDUCTION(VectorBinding, ScalarBinding, module, "mean",
        mean, "Compute the arithmetic mean.")
    AUTODIFF_PYTHON_DEF_REDUCTION(
        VectorBinding, ScalarBinding, module, "norm", norm, "L²-norm.")
    AUTODIFF_PYTHON_DEF_REDUCTION(VectorBinding, ScalarBinding, module,
        "squared_norm", squaredNorm,
        R"doc(Squared L²-norm.

Equal to the dot product of the vector with itself.)doc");
    AUTODIFF_PYTHON_DEF_REDUCTION(VectorBinding, ScalarBinding, module, "sum",
        total, "Sum of vector elements.")

    // matrix reductions

    AUTODIFF_PYTHON_DEF_REDUCTION(MatrixBinding, ScalarBinding, module, "mean",
        mean, "Compute the arithmetic mean.")
    AUTODIFF_PYTHON_DEF_REDUCTION(MatrixBinding, ScalarBinding, module, "norm",
        norm, "Frobenius norm (L²).")
    AUTODIFF_PYTHON_DEF_REDUCTION(MatrixBinding, ScalarBinding, module,
        "squared_norm", squaredNorm,
        R"doc(Squared Frobenius norm (L²).

Equal to the dot product of the matrix with itself.)doc");
    AUTODIFF_PYTHON_DEF_REDUCTION(MatrixBinding, ScalarBinding, module, "sum",
        total, "Sum of matrix elements.")
}
