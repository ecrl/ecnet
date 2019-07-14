import unittest
from os import remove
from os.path import join

import ecnet.utils.data_utils as data_utils


DB_LOC = 'cn_model_v1.0.csv'


class TestDataUtils(unittest.TestCase):

    def test_df_init(self):

        print('\nUNIT TEST: DataFrame init')
        df = data_utils.DataFrame(DB_LOC)
        self.assertEqual(len(df._string_names), 7)
        self.assertEqual(len(df._group_names), 1)
        self.assertEqual(len(df._target_names), 1)
        self.assertEqual(len(df._input_names), 15)
        self.assertEqual(len(df), 482)

    def test_set_creation(self):

        print('\nUNIT TEST: DataFrame set creation')
        df = data_utils.DataFrame(DB_LOC)

        df.create_sets()
        self.assertEqual(len(df.learn_set), 329)
        self.assertEqual(len(df.valid_set), 118)
        self.assertEqual(len(df.test_set), 35)

        df.create_sets(random=True, split=[0.7, 0.2, 0.1])
        self.assertEqual(len(df.learn_set), 337)
        self.assertEqual(len(df.valid_set), 96)
        self.assertEqual(len(df.test_set), 49)

    def test_normalize(self):

        print('\nUNIT TEST: DataFrame normalize')
        df = data_utils.DataFrame(DB_LOC)
        df.normalize()
        df.create_sets(random=True)
        pd = df.package_sets()
        for entry in pd.learn_x:
            for val in entry:
                self.assertGreaterEqual(val, 0)
                self.assertLessEqual(val, 1)
        for entry in pd.valid_x:
            for val in entry:
                self.assertGreaterEqual(val, 0)
                self.assertLessEqual(val, 1)
        for entry in pd.test_x:
            for val in entry:
                self.assertGreaterEqual(val, 0)
                self.assertLessEqual(val, 1)

    def test_shuffle(self):

        print('\nUNIT TEST: DataFrame shuffle')
        df = data_utils.DataFrame(DB_LOC)

        df.shuffle(sets='all', split=[0.7, 0.2, 0.1])
        self.assertEqual(len(df.learn_set), 337)
        self.assertEqual(len(df.valid_set), 96)
        self.assertEqual(len(df.test_set), 49)

        df.shuffle(sets='train', split=[0.7, 0.2, 0.1])
        self.assertEqual(len(df.learn_set), 337)
        self.assertEqual(len(df.valid_set), 96)
        self.assertEqual(len(df.test_set), 49)

    def test_package_sets(self):

        print('\nUNIT TEST: DataFrame package_sets')
        df = data_utils.DataFrame(DB_LOC)
        df.shuffle(sets='all', split=[0.7, 0.2, 0.1])

        pd = df.package_sets()
        self.assertEqual(len(pd.learn_x), 337)
        self.assertEqual(len(pd.valid_x), 96)
        self.assertEqual(len(pd.test_x), 49)
        for entry in pd.learn_x:
            self.assertEqual(len(entry), 15)
        for entry in pd.valid_x:
            self.assertEqual(len(entry), 15)
        for entry in pd.test_x:
            self.assertEqual(len(entry), 15)
        self.assertEqual(len(pd.learn_y), 337)
        self.assertEqual(len(pd.valid_y), 96)
        self.assertEqual(len(pd.test_y), 49)
        for entry in pd.learn_y:
            self.assertEqual(len(entry), 1)
        for entry in pd.valid_y:
            self.assertEqual(len(entry), 1)
        for entry in pd.test_y:
            self.assertEqual(len(entry), 1)

    def test_set_inputs(self):

        print('\nUNIT TEST: DataFrame set_inputs')
        df = data_utils.DataFrame(DB_LOC)
        df.set_inputs(['PHI', 'piPC05'])
        self.assertEqual(len(df._input_names), 2)
        df.create_sets(random=True)
        pd = df.package_sets()
        self.assertEqual(len(pd.learn_x[0]), 2)

    def test_save_df(self):

        print('\nUNIT TEST: DataFrame save')
        df = data_utils.DataFrame(DB_LOC)
        df.save('cn_test_save.csv')
        df_new = data_utils.DataFrame('cn_test_save.csv')
        self.assertEqual(
            len(df),
            len(df_new)
        )
        self.assertEqual(
            df._string_names,
            df_new._string_names
        )
        self.assertEqual(
            df._group_names,
            df_new._group_names
        )
        self.assertEqual(
            df._target_names,
            df_new._target_names
        )
        self.assertEqual(
            df._input_names,
            df_new._input_names
        )
        remove('cn_test_save.csv')


if __name__ == '__main__':

    DB_LOC = join('../', 'cn_model_v1.0.csv')
    unittest.main()
