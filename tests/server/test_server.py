import unittest
from os.path import exists, isdir, join
from os import remove
from shutil import rmtree

from ecnet import Server
from ecnet.utils.server_utils import default_config
from ecnet.utils.logging import logger
from ecnet.utils.data_utils import DataFrame, PackagedData


class TestServer(unittest.TestCase):

    def test_init(self):

        print('\nUNIT TEST: Server init')
        sv = Server()
        self.assertTrue(exists('config.yml'))
        self.assertEqual(sv._vars, default_config())
        remove('config.yml')

    def test_load_data(self):

        print('\nUNIT TEST: Server.load_data')
        sv = Server()
        sv.load_data('cn_model_v1.0.csv')
        self.assertEqual(len(sv._df), 482)
        self.assertEqual(type(sv._sets), PackagedData)
        remove('config.yml')

    def test_create_project(self):

        print('\nUNIT TEST: Server.create_project')
        sv = Server()
        sv.create_project('test_project', 3, 5)
        for pool in range(3):
            for candidate in range(5):
                self.assertTrue(isdir(join(
                    'test_project',
                    'pool_{}'.format(pool),
                    'candidate_{}'.format(candidate)
                )))
        remove('config.yml')
        rmtree('test_project')

    def test_train_project(self):

        print('\nUNIT TEST: Server.train')
        sv = Server()
        sv.load_data('cn_model_v1.0.csv', random=True, split=[0.7, 0.2, 0.1])
        sv.create_project('test_project', 2, 2)
        sv._vars['epochs'] = 100
        sv.train()
        for pool in range(2):
            self.assertTrue(exists(join(
                'test_project',
                'pool_{}'.format(pool),
                'model.h5'
            )))
            for candidate in range(2):
                self.assertTrue(exists(join(
                    'test_project',
                    'pool_{}'.format(pool),
                    'candidate_{}'.format(candidate),
                    'model.h5'
                )))
        remove('config.yml')
        rmtree('test_project')

    def test_use_project(self):

        print('\nUNIT TEST: Server.use')
        sv = Server()
        sv.load_data('cn_model_v1.0.csv', random=True, split=[0.7, 0.2, 0.1])
        sv.create_project('test_project', 2, 2)
        sv._vars['epochs'] = 100
        sv.train()
        results = sv.use()
        self.assertEqual(len(results), len(sv._df))
        remove('config.yml')
        rmtree('test_project')

    def test_save_project(self):

        print('\nUNIT TEST: Server.save_project')
        sv = Server()
        sv.load_data('cn_model_v1.0.csv', random=True, split=[0.7, 0.2, 0.1])
        sv.create_project('test_project', 2, 2)
        sv._vars['epochs'] = 100
        sv.train()
        sv.save_project()
        self.assertTrue(exists('test_project.prj'))
        self.assertTrue(not isdir('test_project'))
        remove('test_project.prj')
        remove('config.yml')

    def test_multiprocessing_train(self):

        sv = Server(num_processes=8)
        sv.load_data('cn_model_v1.0.csv')
        sv.create_project('test_project', 2, 4)
        sv._vars['epochs'] = 100
        sv.train()
        for pool in range(2):
            self.assertTrue(exists(join(
                'test_project',
                'pool_{}'.format(pool),
                'model.h5'
            )))
            for candidate in range(4):
                self.assertTrue(exists(join(
                    'test_project',
                    'pool_{}'.format(pool),
                    'candidate_{}'.format(candidate),
                    'model.h5'
                )))
        remove('config.yml')
        rmtree('test_project')


if __name__ == '__main__':

    unittest.main()
