import unittest
from os import remove
from os.path import join

from ecnet.utils.data_utils import DataFrame
from ecnet.tasks.limit_inputs import limit_rforest
from ecnet import Server


DB_LOC = 'cn_model_v1.0.csv'


class TestLimit(unittest.TestCase):

    def test_limit_rforest(self):

        print('\nUNIT TEST: limit_rforest')
        df = DataFrame(DB_LOC)
        df.create_sets()
        result = limit_rforest(df, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        df.set_inputs([r[0] for r in result])
        self.assertEqual(len(df._input_names), 2)

    def test_limit_sets(self):

        print('\nUNIT TEST: limit_sets')
        df = DataFrame(DB_LOC)
        df.create_sets()
        result = limit_rforest(df, 2, eval_set='valid')
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        df.set_inputs([r[0] for r in result])
        self.assertEqual(len(df._input_names), 2)
        df = DataFrame(DB_LOC)
        df.create_sets()
        result = limit_rforest(df, 2, eval_set='train')
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        df.set_inputs([r[0] for r in result])
        self.assertEqual(len(df._input_names), 2)
        df = DataFrame(DB_LOC)
        df.create_sets()
        result = limit_rforest(df, 2, eval_set='test')
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        df.set_inputs([r[0] for r in result])
        self.assertEqual(len(df._input_names), 2)
        df = DataFrame(DB_LOC)
        df.create_sets()
        result = limit_rforest(df, 2, eval_set=None)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        df.set_inputs([r[0] for r in result])
        self.assertEqual(len(df._input_names), 2)

    def test_server_limit(self):

        print('\nUNIT TEST: limit_rforest (Server)')
        sv = Server()
        sv.load_data(DB_LOC)
        sv.limit_inputs(2, output_filename='cn_limited.csv')
        self.assertEqual(len(sv._df._input_names), 2)
        self.assertEqual(len(sv._sets.learn_x[0]), 2)
        sv.load_data('cn_limited.csv')
        self.assertEqual(len(sv._df._input_names), 2)
        self.assertEqual(len(sv._sets.learn_x[0]), 2)
        remove('cn_limited.csv')
        remove('config.yml')


if __name__ == '__main__':

    DB_LOC = join('../', 'cn_model_v1.0.csv')
    unittest.main()
