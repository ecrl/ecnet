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

        with open('smiles.smi', 'w') as smi_file:
            smi_file.write('CCC')
        smi_file.close()
        predict('smiles.smi', 'results.csv', 'test_project', form='smiles')
        with open('results.csv', 'r') as res_file:
            self.assertGreater(len(res_file.read()), 0)
        res_file.close()
        remove('smiles.smi')
        remove('results.csv')
        remove('test_project.prj')
        remove('config.yml')


if __name__ == '__main__':

    unittest.main()
