from ecnet.tools.database import create_db


def from_names():

    print('Creating database from names...')
    create_db(
        'mols_names.txt',
        'db_from_names.csv',
        targets='mols_targets.txt'
    )


def from_smiles():

    print('Creating database from SMILES...')
    create_db(
        'mols_smiles.txt',
        'db_from_smiles.csv',
        targets='mols_targets.txt',
        form='smiles'
    )


def fingerprints():

    print('Creating database with fingerprints...')
    create_db(
        'mols_smiles.txt',
        'db_with_fingerprints.csv',
        targets='mols_targets.txt',
        form='smiles',
        fingerprints=True
    )


if __name__ == '__main__':

    from_names()
    from_smiles()
    fingerprints()
