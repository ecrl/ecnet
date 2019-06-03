import unittest
from os import remove

from ecnet.utils.data_utils import DataFrame
import ecnet.tools.database as database


class TestDatabase(unittest.TestCase):

    def test_create_db(self):

        print('\nUNIT TEST: create_db')
        with open('smiles.smi', 'w') as smi_file:
            smi_file.write('CCC')
        smi_file.close()
        database.create_db('smiles.smi', 'database.csv', form='smiles')
        df = DataFrame('database.csv')
        self.assertEqual(len(df), 1)
        self.assertEqual(len(df._input_names), 1875)
        remove('smiles.smi')
        remove('database.csv')


if __name__ == '__main__':

    unittest.main()
