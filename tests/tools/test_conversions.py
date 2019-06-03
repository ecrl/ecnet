import unittest
from os import remove
from csv import DictReader

import ecnet.tools.conversions as conversions


class TestConversions(unittest.TestCase):

    def test_get_smiles(self):

        print('\nUNIT TEST: get_smiles')
        smiles = conversions.get_smiles('Propane')
        self.assertIn('CCC', smiles)

    def test_smiles_to_descriptors(self):

        print('\nUNIT TEST: smiles_to_descriptors')
        with open('smiles.smi', 'w') as smi_file:
            smi_file.write('CCC')
        smi_file.close()
        conversions.smiles_to_descriptors('smiles.smi', 'desc.csv')
        with open('desc.csv', 'r') as desc_file:
            reader = DictReader(desc_file)
            for row in reader:
                mol_row = row
            self.assertEqual(len(list(mol_row.keys())), 1876)
        desc_file.close()
        remove('smiles.smi')
        remove('desc.csv')

    def test_smiles_to_mdl(self):

        print('\nUNIT TEST: smiles_to_mdl')
        with open('smiles.smi', 'w') as smi_file:
            smi_file.write('CCC')
        smi_file.close()
        conversions.smiles_to_mdl('smiles.smi', 'mdl.mdl')
        with open('mdl.mdl', 'r') as mdl_file:
            self.assertGreater(len(mdl_file.read()), 0)
        mdl_file.close()
        remove('smiles.smi')
        remove('mdl.mdl')

    def test_mdl_to_descriptors(self):

        print('\nUNIT TEST: mdl_to_descriptors')
        with open('smiles.smi', 'w') as smi_file:
            smi_file.write('CCC')
        smi_file.close()
        conversions.smiles_to_mdl('smiles.smi', 'mdl.mdl')
        conversions.mdl_to_descriptors('mdl.mdl', 'desc.csv')
        with open('desc.csv', 'r') as desc_file:
            reader = DictReader(desc_file)
            for row in reader:
                mol_row = row
            self.assertEqual(len(list(mol_row.keys())), 1876)
        desc_file.close()
        remove('smiles.smi')
        remove('mdl.mdl')
        remove('desc.csv')


if __name__ == '__main__':

    unittest.main()
