import unittest
from os import remove

from ecnet.utils.data_utils import DataFrame
import ecnet.tools.database as database


class TestDatabase(unittest.TestCase):

    def test_create_db(self):

        print('\nUNIT TEST: create_db')

        database.create_db(['CCC', 'CC'], 'database.csv')
        df = DataFrame('database.csv')
        self.assertEqual(len(df), 2)
        self.assertEqual(len(df._input_names), 1875)
        remove('database.csv')

    def test_create_db_convert(self):

        print('\nUNIT TEST: create_db, convert MDL')

        database.create_db(['CCC', 'CC'], 'database.csv', convert_mdl=True)
        df = DataFrame('database.csv')
        self.assertEqual(len(df), 2)
        self.assertEqual(len(df._input_names), 1875)
        remove('database.csv')


if __name__ == '__main__':

    unittest.main()
