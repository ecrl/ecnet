import unittest

from ecnet.utils.data_utils import DataFrame
from ecnet.utils.server_utils import default_config
from ecnet.tasks.tuning import tune_hyperparameters


class TestTune(unittest.TestCase):

    def test_tune_hyperparameters(self):

        print('\nUNIT TEST: tune_hyperparameters')
        df = DataFrame('cn_model_v1.0.csv')
        df.create_sets(random=True)
        config = default_config()
        new_hp = tune_hyperparameters(df, config, 2, 1)
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
        self.assertLessEqual(new_hp['hidden_layers'][0][0], 100)
        self.assertGreaterEqual(new_hp['hidden_layers'][1][0], 1)
        self.assertLessEqual(new_hp['hidden_layers'][1][0], 100)


if __name__ == '__main__':

    unittest.main()
