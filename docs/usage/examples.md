# Example Scripts

## SMILES String Validation

SMILES strings are the basis for QSPR descriptor generation, and therefore play an immense role in what neural networks learn (and how they correlate QSPR descriptors to given fuel properties). It is paramount that SMILES strings for molecules are correct to ensure neural networks learn from correct molecule representations.

To validate SMILES strings for molecules stored in an ECNet-formatted database, we can use the script below to query PubChem using molecule names. The "validate_smiles" function accepts two arguments, the database you wish to validate and the filename of the resulting validated database. Note that QSPR descriptors in the resulting database do not reflect changes made to SMILES strings, and you will need to create a new database using our [database construction tool](https://ecnet.readthedocs.io/en/latest/usage/tools.html#database-creation) to generate new descriptors.

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Example script for validating ECNet-formatted database SMILES strings
#

from ecnet.utils.data_utils import DataFrame
from ecnet.tools.database import get_smiles
from ecnet.utils.logging import logger


def validate_smiles(db_name, new_db):

    # load the database
    logger.log('info', 'Loading data from {}'.format(db_name))
    df = DataFrame(db_name)

    # find index of `Compound Name` string
    name_idx = -1
    for idx, name in enumerate(df.string_names):
        if name == 'Compound Name':
            name_idx = idx
            break
    if name_idx == -1:
        logger.log('error', '`Compound Name` string not found in database')
        return

    # find index of `SMILES` string
    smiles_idx = -1
    for idx, name in enumerate(df.string_names):
        if name == 'SMILES':
            smiles_idx = idx
            break
    if smiles_idx == -1:
        logger.log('error', '`SMILES` string not found in database')

    # check each molecule's SMILES, replace if incorrect
    for pt in df.data_points:
        smiles = get_smiles(pt.strings[name_idx])
        if smiles == '':
            logger.log('warn', '{} not found on PubChem'.format(
                pt.strings[name_idx]
            ))
            continue
        else:
            if smiles != pt.strings[smiles_idx]:
                logger.log(
                    'crit',
                    'Incorrect SMILES for {}:\n\tDatabase SMILES: {}'
                    '\n\tPubChem SMILES: {}'.format(
                        pt.strings[name_idx],
                        pt.strings[smiles_idx],
                        smiles
                    ))
                pt.strings[smiles_idx] = smiles
            else:
                logger.log('info', 'Correct SMILES for {}'.format(
                    pt.strings[name_idx]
                ))

    # save the validated database
    logger.log('info', 'Saving validated data to {}'.format(new_db))
    df.save(new_db)
    return


if __name__ == '__main__':

    # initialize logging
    logger.stream_level = 'info'
    # un-comment this for file logging
    # logger.file_level = 'info'

    validate_smiles('unvalidated_db.csv', 'validated_db.csv')

```
