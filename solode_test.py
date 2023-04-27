import unittest
import numpy as np
import solode  

class TestODESolver(unittest.TestCase):

    def test_f(self):
        x = 2
        t = 3
        args = {}
        result = solode.f(x, t, args)
        self.assertEqual(result, x)

    def test_solve_ode_dimension_mismatch(self):
        method = 'euler'
        f = solode.f
        t = np.linspace(0, 1, 100)
        X0 = np.array([1, 2])

        with self.assertRaises(ValueError):
            solode.solve_ode(method, f, t, X0)

    def test_solve_ode_unknown_method(self):
        method = 'unknown'
        f = solode.f
        t = np.linspace(0, 1, 100)
        X0 = np.array([1])

        with self.assertRaises(ValueError):
            solode.solve_ode(method, f, t, X0)

    def test_euler_step(self):
        f = solode.f
        x = np.array([1])
        t = 0
        dt = 0.1
        params = {}
        result = solode.euler_step(f, x, t, dt, **params)
        expected_result = np.array([1.1])
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_RK4_step(self):
        f = solode.g
        x = np.array([0, 1])
        t = 0
        dt = 0.1
        params = {}
        result = solode.RK4_step(f, x, t, dt, **params)
        expected_result = np.array([0.09983342, 0.99500417])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=5)


    def test_simpson38_step(self):
        f = solode.f
        x = np.array([1])
        t = 0
        dt = 0.1
        params = {}
        result = solode.simpson38_step(f, x, t, dt, **params)
        expected_result = np.array([1.10517092])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=5)

    def test_midpoint_step(self):
        f = solode.f
        x = np.array([1])
        t = 0
        dt = 0.1
        params = {}
        result = solode.midpoint_step(f, x, t, dt, **params)
        expected_result = np.array([1.1])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=5)
if __name__ == '__main__':
    unittest.main()
