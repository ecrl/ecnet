import unittest
from copy import deepcopy

from ecnet.utils.data_utils import DataFrame
from ecnet.tasks.remove_outliers import remove_outliers


class TestRemoveOutliers(unittest.TestCase):

    def test_limit_rforest(self):

        print('UNIT TEST: remove_outliers')
        df = DataFrame('cn_model_v1.0.csv')
        df_res = remove_outliers(deepcopy(df))
        self.assertLess(len(df_res), len(df))


if __name__ == '__main__':

    unittest.main()
