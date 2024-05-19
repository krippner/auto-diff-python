import unittest
from autodiff.scalar import Function, var, d

class TestScalarProduct(unittest.TestCase):
    def test_eager_evaluation(self):
        xVal = 0.5
        yVal = -2.5

        x = var(xVal)
        y = var(yVal)
        z = var(x * y)

        assert x() == xVal
        assert y() == yVal
        assert z() == xVal * yVal

    def test_lazy_evaluation(self):
        xVal = 0.5
        yVal = -2.5

        x = var(xVal)
        y = var(yVal)
        z = var(x * y)
        
        xNewVal = 3.5
        yNewVal = 1.0

        x.set(xNewVal)
        y.set(yNewVal)


        f = Function(z)
        f.evaluate()

        assert x() == xNewVal
        assert y() == yNewVal
        assert z() == xNewVal * yNewVal

    def test_reverse_mode_differentiation(self):
        xVal = 0.5
        yVal = -2.5

        x = var(xVal)
        y = var(yVal)
        z = var(x * y)

        f = Function(z)
        f.pull_gradient_at(z)

        assert d(x) == yVal
        assert d(y) == xVal
        assert d(z) == 1.0

if __name__ == '__main__':
    unittest.main()
