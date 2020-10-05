import unittest
from os.path import join

from ecnet.utils.data_utils import DataFrame
from ecnet.utils.server_utils import default_config
from ecnet.tasks.tuning import tune_hyperparameters


DB_LOC = 'cn_model_v1.0.csv'


class TestTune(unittest.TestCase):

    def test_tune_hyperparameters(self):

        print('\nUNIT TEST: tune_hyperparameters')
        df = DataFrame(DB_LOC)
        df.create_sets(random=True)
        config = default_config()
        new_hp = tune_hyperparameters(df, config, 2, 1, epochs=100)
        self.assertGreaterEqual(new_hp['beta_1'], 0)
        self.assertLessEqual(new_hp['beta_1'], 1)
        self.assertGreaterEqual(new_hp['beta_2'], 0)
        self.assertLessEqual(new_hp['beta_2'], 1)
        self.assertGreaterEqual(new_hp['decay'], 0)
        self.assertLessEqual(new_hp['decay'], 1)
        self.assertGreaterEqual(new_hp['epsilon'], 0)
        self.assertLessEqual(new_hp['epsilon'], 1)
        self.assertGreaterEqual(new_hp['learning_rate'], 0)
        self.assertLessEqual(new_hp['learning_rate'], 1)
        self.assertGreaterEqual(new_hp['batch_size'], 1)
        self.assertLessEqual(new_hp['batch_size'], len(df.learn_set))
        self.assertGreaterEqual(new_hp['hidden_layers'][0][0], 1)
        self.assertLessEqual(new_hp['hidden_layers'][0][0], 600)
        self.assertGreaterEqual(new_hp['hidden_layers'][1][0], 1)
        self.assertLessEqual(new_hp['hidden_layers'][1][0], 600)

    def test_th_multiprocess(self):

        print('\nUNIT TEST: tune_hyperparameters multiprocessed')
        df = DataFrame(DB_LOC)
        df.create_sets(random=True)
        config = default_config()
        new_hp = tune_hyperparameters(df, config, 2, 1, 2, epochs=100)
        self.assertGreaterEqual(new_hp['beta_1'], 0)
        self.assertLessEqual(new_hp['beta_1'], 1)
        self.assertGreaterEqual(new_hp['beta_2'], 0)
        self.assertLessEqual(new_hp['beta_2'], 1)
        self.assertGreaterEqual(new_hp['decay'], 0)
        self.assertLessEqual(new_hp['decay'], 1)
        self.assertGreaterEqual(new_hp['epsilon'], 0)
        self.assertLessEqual(new_hp['epsilon'], 1)
        self.assertGreaterEqual(new_hp['learning_rate'], 0)
        self.assertLessEqual(new_hp['learning_rate'], 1)
        self.assertGreaterEqual(new_hp['batch_size'], 1)
        self.assertLessEqual(new_hp['batch_size'], len(df.learn_set))
        self.assertGreaterEqual(new_hp['hidden_layers'][0][0], 1)
        self.assertLessEqual(new_hp['hidden_layers'][0][0], 600)
        self.assertGreaterEqual(new_hp['hidden_layers'][1][0], 1)
        self.assertLessEqual(new_hp['hidden_layers'][1][0], 600)


if __name__ == '__main__':

    DB_LOC = join('../', 'cn_model_v1.0.csv')
    unittest.main()
