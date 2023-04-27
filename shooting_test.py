import unittest
import numpy as np
from shootfinal import predator_prey, pc_predator_prey, numerical_shooting

class TestShooting(unittest.TestCase):

    def test_predator_prey(self):
        X = [0.5, 0.5]
        t = 0
        params = {'a': 1, 'b': 0.2, 'd': 0.1}
        result = predator_prey(X, t, params)
        expected = [-0.2, 0.1]
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_pc_predator_prey(self):
        X0 = [0.5, 0.5]
        params = {'a': 1, 'b': 0.2, 'd': 0.1}
        result = pc_predator_prey(X0, **params)
        expected = -0.2
        self.assertAlmostEqual(result, expected, places=5)

    def test_numerical_shooting(self):
        X0 = [1.3, 1.3]
        T_guess = 10
        params = {'a': 1, 'b': 0.2, 'd': 0.1}
        result_X0, result_T = numerical_shooting(X0, T_guess, predator_prey, pc_predator_prey, **params)
        expected_X0 = [1.28573934, 1.28573934]
        expected_T = 6.69709213
        np.testing.assert_almost_equal(result_X0, expected_X0, decimal=5)
        self.assertAlmostEqual(result_T, expected_T, places=5)

    def test_numerical_shooting_invalid_dimensions(self):
        X0 = [1.3, 1.3, 1.3] # Invalid dimensions
        T_guess = 10
        params = {'a': 1, 'b': 0.2, 'd': 0.1}
        with self.assertRaises(ValueError):
            numerical_shooting(X0, T_guess, predator_prey, pc_predator_prey, **params)

    def test_numerical_shooting_non_converging(self):
        X0 = [1.3, 1.3]
        T_guess = -1 # Invalid T_guess causing non-convergence
        params = {'a': 1, 'b': 0.2, 'd': 0.1}
        result_X0, result_T = numerical_shooting(X0, T_guess, predator_prey, pc_predator_prey, **params)
        self.assertEqual(result_X0, [])
        self.assertEqual(result_T, [])

if __name__ == '__main__':
    unittest.main()

