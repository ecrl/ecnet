import unittest

from ecnet.utils.data_utils import DataFrame
from ecnet.tasks.limit_inputs import limit_rforest


class TestLimit(unittest.TestCase):

    def test_limit_rforest(self):

        print('\nUNIT TEST: limit_rforest')
        df = DataFrame('cn_model_v1.0.csv')
        df_res = limit_rforest(df, 2)
        self.assertEqual(len(df._input_names), 2)
        self.assertIn(df_res._input_names[0], df._input_names)
        self.assertIn(df_res._input_names[1], df._input_names)


if __name__ == '__main__':

    unittest.main()
