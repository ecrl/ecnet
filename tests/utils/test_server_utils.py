import unittest
from os.path import exists, join
from os import remove

from numpy import array

import ecnet.utils.server_utils as server_utils
import ecnet.utils.data_utils as data_utils


DB_LOC = 'cn_model_v1.0.csv'


class TestServerUtils(unittest.TestCase):

    def test_check_config(self):

        print('\nUNIT TEST: check_config')
        dc = server_utils.default_config()
        del dc['batch_size']
        self.assertFalse('batch_size' in list(dc.keys()))
        dc = server_utils.check_config(dc)
        self.assertTrue('batch_size' in list(dc.keys()))
        self.assertEqual(dc['batch_size'], 32)

    def test_default_config(self):

        print('\nUNIT TEST: default_config')
        dc = server_utils.default_config()
        self.assertEqual(
            dc, {
                'epochs': 3000,
                'learning_rate': 0.01,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-8,
                'decay': 0.0,
                'hidden_layers': [
                    [32, 'relu'],
                    [32, 'relu']
                ],
                'output_activation': 'linear',
                'batch_size': 32
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
            join('test_project', 'pool_1', 'candidate_2', 'model.ecnet')
        )
        self.assertEqual(
            server_utils.get_candidate_path(prj_name, pool, candidate,
                                            p_best=True),
            join('test_project', 'pool_1', 'model.ecnet')
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
        df = data_utils.DataFrame(DB_LOC)
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
        df = data_utils.DataFrame(DB_LOC)
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
                'epochs': 3000,
                'learning_rate': 0.01,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-8,
                'decay': 0.0,
                'hidden_layers': [
                    [32, 'relu'],
                    [32, 'relu']
                ],
                'output_activation': 'linear',
                'batch_size': 32
            }
        )
        remove('config.yml')

    def test_open_save_df(self):

        print('\nUNIT TEST: open/save DataFrame')
        df = data_utils.DataFrame(DB_LOC)
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
        df = data_utils.DataFrame(DB_LOC)
        df.create_sets(random=True)
        pd = df.package_sets()
        config = server_utils.default_config()
        config['epochs'] = 100
        r_squared = server_utils.train_model(
            pd, config, 'test', 'r2', filename='test_train.ecnet'
        )
        self.assertTrue(exists('test_train.ecnet'))
        remove('test_train.ecnet')

    def test_use_model(self):

        print('\nUNIT TEST: use_model')
        df = data_utils.DataFrame(DB_LOC)
        df.create_sets(random=True)
        pd = df.package_sets()
        config = server_utils.default_config()
        config['epochs'] = 100
        _ = server_utils.train_model(
            pd, config, 'test', 'rmse', filename='test_use.ecnet'
        )
        self.assertEqual(
            len(server_utils.use_model(pd, 'learn', 'test_use.ecnet')),
            len(pd.learn_y)
        )
        self.assertEqual(
            len(server_utils.use_model(pd, 'valid', 'test_use.ecnet')),
            len(pd.valid_y)
        )
        self.assertEqual(
            len(server_utils.use_model(pd, 'test', 'test_use.ecnet')),
            len(pd.test_y)
        )
        self.assertEqual(
            len(server_utils.use_model(pd, 'train', 'test_use.ecnet')),
            len(pd.learn_y) + len(pd.valid_y)
        )
        self.assertEqual(
            len(server_utils.use_model(pd, None, 'test_use.ecnet')),
            len(pd.learn_y) + len(pd.valid_y) + len(pd.test_y)
        )
        remove('test_use.ecnet')


if __name__ == '__main__':

    DB_LOC = join('../', 'cn_model_v1.0.csv')
    unittest.main()
