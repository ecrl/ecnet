import unittest
from os import remove

from ecnet import Server
from ecnet.tools.project import predict


class TestUseProject(unittest.TestCase):

    def test_predict(self):

        print('\nUNIT TEST: project.predict')
        sv = Server()
        sv.load_data('cn_model_v2.0.csv')
        sv.create_project('test_project', 1, 1)
        sv.train()
        sv.save_project()

        results = predict(['CCC', 'CCCC'], 'test_project.prj', 'results.csv')
        print(results)

        self.assertEqual(len(results), 2)
        with open('results.csv', 'r') as res_file:
            self.assertGreater(len(res_file.read()), 0)
        res_file.close()

        remove('results.csv')
        remove('test_project.prj')
        remove('config.yml')


if __name__ == '__main__':

    unittest.main()
