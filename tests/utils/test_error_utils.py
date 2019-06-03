import unittest

from numpy import array

import ecnet.utils.error_utils as error_utils


class TestErrors(unittest.TestCase):

    def test_rmse(self):

        print('\nUNIT TEST: RMSE')
        y = array([1, 2, 3, 4, 5])
        y_hat = array([2, 4, 6, 8, 10])
        self.assertEqual(round(error_utils.calc_rmse(y_hat, y), 3), 3.317)

    def test_mean_abs_error(self):

        print('\nUNIT TEST: Mean Absolute Error')
        y = array([1, 2, 3, 4, 5])
        y_hat = array([2, 4, 6, 8, 10])
        self.assertEqual(error_utils.calc_mean_abs_error(y_hat, y), 3)

    def test_med_abs_error(self):

        print('\nUNIT TEST: Median Absolute Error')
        y = array([1, 2, 3, 4, 5])
        y_hat = array([2, 4, 6, 8, 10])
        self.assertEqual(error_utils.calc_med_abs_error(y_hat, y), 3)

    def test_r2(self):

        print('\nUNIT TEST: R-Squared')
        y = array([1, 2, 3, 4, 5])
        y_hat = array([1, 2, 3, 4, 6])
        self.assertEqual(error_utils.calc_r2(y_hat, y), 0.9)


if __name__ == '__main__':

    unittest.main()
