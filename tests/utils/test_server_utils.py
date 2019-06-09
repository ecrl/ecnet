import unittest
from os.path import exists, join
from os import remove

from numpy import array

import ecnet.utils.server_utils as server_utils
import ecnet.utils.data_utils as data_utils


class TestServerUtils(unittest.TestCase):

    def test_default_config(self):

        print('\nUNIT TEST: default_config')
        dc = server_utils.default_config()
        self.assertEqual(
            dc, {
                'epochs': 10000,
                'learning_rate': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 0.0000001,
                'decay': 0.0,
                'hidden_layers': [
                    [32, 'relu'],
                    [32, 'relu']
                ],
                'output_activation': 'linear'
            }
        )

    def test_get_candidate_path(self):

        print('\nUNIT TEST: get_candidate_path')
        prj_name = 'test_project'
        pool = 1
        candidate = 2
        self.assertEqual(
            server_utils.get_candidate_path(prj_name, pool, candidate),
            join('test_project', 'pool_1', 'candidate_2')
        )
        self.assertEqual(
            server_utils.get_candidate_path(prj_name, pool, candidate, True),
            join('test_project', 'pool_1', 'candidate_2', 'model.h5')
        )
        self.assertEqual(
            server_utils.get_candidate_path(prj_name, pool, candidate,
                                            p_best=True),
            join('test_project', 'pool_1', 'model.h5')
        )

    def test_get_error(self):

        print('\nUNIT TEST: get_error')
        y = array([1, 2, 3, 4, 5])
        y_hat = array([2, 4, 6, 8, 10])
        self.assertEqual(
            round(server_utils.get_error(y_hat, y, 'rmse'), 3),
            3.317
        )
        self.assertEqual(server_utils.get_error(y_hat, y, 'mean_abs_error'), 3)
        self.assertEqual(server_utils.get_error(y_hat, y, 'med_abs_error'), 3)
        self.assertEqual(
            server_utils.get_error(array([1, 2, 3, 4, 6]), y, 'r2'),
            0.9
        )

    def test_get_x(self):

        print('\nUNIT TEST: get_x')
        df = data_utils.DataFrame('cn_model_v1.0.csv')
        df.create_sets(random=True)
        pd = df.package_sets()
        self.assertEqual(len(server_utils.get_x(pd, 'learn')), len(pd.learn_x))
        self.assertEqual(len(server_utils.get_x(pd, 'valid')), len(pd.valid_x))
        self.assertEqual(len(server_utils.get_x(pd, 'test')), len(pd.test_x))
        self.assertEqual(
            len(server_utils.get_x(pd, 'train')),
            len(pd.learn_x) + len(pd.valid_x)
        )
        self.assertEqual(
            len(server_utils.get_x(pd, None)),
            len(pd.learn_x) + len(pd.valid_x) + len(pd.test_x)
        )

    def test_get_y(self):

        print('\nUNIT TEST: get_y')
        df = data_utils.DataFrame('cn_model_v1.0.csv')
        df.create_sets(random=True)
        pd = df.package_sets()
        self.assertEqual(len(server_utils.get_y(pd, 'learn')), len(pd.learn_y))
        self.assertEqual(len(server_utils.get_y(pd, 'valid')), len(pd.valid_y))
        self.assertEqual(len(server_utils.get_y(pd, 'test')), len(pd.test_y))
        self.assertEqual(
            len(server_utils.get_y(pd, 'train')),
            len(pd.learn_y) + len(pd.valid_y)
        )
        self.assertEqual(
            len(server_utils.get_y(pd, None)),
            len(pd.learn_y) + len(pd.valid_y) + len(pd.test_y)
        )

    def test_open_save_config(self):

        print('\nUNIT TEST: open/save config')
        config = server_utils.default_config()
        server_utils.save_config(config, 'config.yml')
        config = server_utils.open_config('config.yml')
        self.assertEqual(
            server_utils.open_config('config.yml'),
            {
                'epochs': 10000,
                'learning_rate': 0.001,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 0.0000001,
                'decay': 0.0,
                'hidden_layers': [
                    [32, 'relu'],
                    [32, 'relu']
                ],
                'output_activation': 'linear'
            }
        )
        remove('config.yml')

    def test_open_save_df(self):

        print('\nUNIT TEST: open/save DataFrame')
        df = data_utils.DataFrame('cn_model_v1.0.csv')
        server_utils.save_df(df, 'test_df_saved.d')
        df = server_utils.open_df('test_df_saved.d')
        self.assertEqual(len(df), 482)
        self.assertEqual(len(df._string_names), 7)
        self.assertEqual(len(df._group_names), 1)
        self.assertEqual(len(df._target_names), 1)
        self.assertEqual(len(df._input_names), 15)
        remove('test_df_saved.d')

    def test_train_model(self):

        print('\nUNIT TEST: train_model')
        df = data_utils.DataFrame('cn_model_v1.0.csv')
        df.create_sets(random=True)
        pd = df.package_sets()
        config = server_utils.default_config()
        r_squared = server_utils.train_model(
            pd, config, 'test', 'r2', filename='test_train.h5'
        )
        self.assertTrue(exists('test_train.h5'))
        self.assertGreaterEqual(r_squared, 0)
        self.assertLessEqual(r_squared, 1)
        remove('test_train.h5')

    def test_use_model(self):

        print('\nUNIT TEST: use_model')
        df = data_utils.DataFrame('cn_model_v1.0.csv')
        df.create_sets(random=True)
        pd = df.package_sets()
        config = server_utils.default_config()
        _ = server_utils.train_model(
            pd, config, 'test', 'rmse', filename='test_use.h5'
        )
        self.assertEqual(
            len(server_utils.use_model(pd, 'learn', 'test_use.h5')),
            len(pd.learn_y)
        )
        self.assertEqual(
            len(server_utils.use_model(pd, 'valid', 'test_use.h5')),
            len(pd.valid_y)
        )
        self.assertEqual(
            len(server_utils.use_model(pd, 'test', 'test_use.h5')),
            len(pd.test_y)
        )
        self.assertEqual(
            len(server_utils.use_model(pd, 'train', 'test_use.h5')),
            len(pd.learn_y) + len(pd.valid_y)
        )
        self.assertEqual(
            len(server_utils.use_model(pd, None, 'test_use.h5')),
            len(pd.learn_y) + len(pd.valid_y) + len(pd.test_y)
        )
        remove('test_use.h5')


if __name__ == '__main__':

    unittest.main()
