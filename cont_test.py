import unittest
import numpy as np
from cont import hopf, modified_hopf, cubic, continuation

class TestContinuation(unittest.TestCase):

    def test_cubic(self):
        params = {'c': 0}
        result = cubic(1, params)
        self.assertEqual(result, 0)

    def test_hopf(self):
        params = {'beta': 1}
        X = [1, 1]
        result = hopf(X, 0, params)
        self.assertEqual(result, [0, 0])

    def test_modified_hopf(self):
        params = {'beta': 1}
        X = [1, 1]
        result = modified_hopf(X, 0, params)
        self.assertEqual(result, [0, 0])

    def test_continuation_wrong_initial_dimensions(self):
        with self.assertRaises(ValueError):
            continuation([1], 'beta', [2, 0], 80, hopf, 'natural-parameter', 'numerical-shooting', phase_condition=None, T_guess=2, beta=1)

    def test_continuation_wrong_numerical_shooting(self):
        with self.assertRaises(ValueError):
            continuation([1, 1], 'beta', [2, 0], 80, hopf, 'natural-parameter', 'wrong-shooting', phase_condition=None, T_guess=2, beta=1)

    def test_continuation_wrong_method(self):
        with self.assertRaises(ValueError):
            continuation([1, 1], 'beta', [2, 0], 80, hopf, 'wrong-method', 'numerical-shooting', phase_condition=None, T_guess=2, beta=1)

if __name__ == '__main__':
    unittest.main()

