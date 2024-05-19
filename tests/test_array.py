import unittest
import numpy as np
from autodiff.array import Function, var, d

class TestArrayProduct(unittest.TestCase):
    def test_eager_evaluation(self):
        xVal = np.array([0.5, 1.0, 2.0])
        yVal = np.array([-2.5, 1.0, 3.0])

        x = var(xVal)
        y = var(yVal)
        z = var(x * y)

        assert np.array_equal(x(), xVal)
        assert np.array_equal(y(), yVal)
        assert np.array_equal(z(), xVal * yVal)

    def test_lazy_evaluation(self):
        xVal = np.array([0.5, 1.0, 2.0])
        yVal = np.array([-2.5, 1.0, 3.0])

        x = var(xVal)
        y = var(yVal)
        z = var(x * y)

        xNewVal = np.array([3.5, -1.0])
        yNewVal = np.array([1.0, -1.0])

        x.set(xNewVal)
        y.set(yNewVal)

        f = Function(z)
        f.evaluate()

        assert np.array_equal(x(), xNewVal)
        assert np.array_equal(y(), yNewVal)
        assert np.array_equal(z(), xNewVal * yNewVal)

    def test_reverse_mode_differentiation(self):
        xVal = np.array([0.5, 1.0, 2.0])
        yVal = np.array([-2.5, 1.0, 3.0])

        x = var(xVal)
        y = var(yVal)
        z = var(x * y)

        f = Function(z)
        f.pull_gradient_at(z)

        assert np.array_equal(d(x), np.diag(yVal))
        assert np.array_equal(d(y), np.diag(xVal))
        assert np.array_equal(d(z), np.identity(3))

if __name__ == '__main__':
    unittest.main()
