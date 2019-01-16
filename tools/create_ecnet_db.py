#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# tools/create_database.py
#
# Creates ECNet-formatted database from molecule names
#

from argparse import ArgumentParser
from sys import argv
from csv import writer, QUOTE_ALL
from os import remove

import name_to_smiles
import smiles_to_qspr


def parse_args():

    ap = ArgumentParser()
    ap.add_argument(
        'input_file',
        type=str,
        help='Path to TXT file with molecule names'
    )
    ap.add_argument(
        'output_file',
        type=str,
        help='Path to resulting CSV database'
    )
    ap.add_argument(
        '--experimental_vals',
        type=str,
        help='Path to TXT file containing experimental values',
        default=None
    )
    ap.add_argument(
        '--id_prefix',
        type=str,
        help='Prefix for molecule ID in resulting CSV database'
    )
    ap.add_argument(
        '--smiles_file',
        type=str,
        help='Path to temporary .smiles file created when processing',
        default='./mols.smiles'
    )
    ap.add_argument(
        '--padel_path',
        type=str,
        help='Path to PaDEL-Descriptor.jar'
        ' (defaults to ..\PaDEL-Descriptor\PaDEL-Descriptor.jar)',
        default=smiles_to_qspr._PADEL_PATH
    )
    ap.add_argument(
        '--model_file',
        type=str,
        help='Path to temporary .mdl file created when processing',
        default='./mols.mdl'
    )
    ap.add_argument(
        '--descriptors_file',
        type=str,
        help='Path to temporary unformatted .csv descriptor file created when'
        ' processing',
        default='./descriptors.csv'
    )
    ap.add_argument(
        '--clean_up',
        type=bool,
        help='If True, cleans up all temporary files created when processing',
        default=True
    )
    return vars(ap.parse_args(argv[1:]))


def main(args):

    # Open names, experimental vals (if supplied) get SMILES strings (if found)
    print('Reading molecule names from {}'.format(args['input_file']))
    names = name_to_smiles.get_names(args['input_file'])
    if args['experimental_vals'] is not None:
        print('Reading experimental values from {}'.format(
            args['experimental_vals'])
        )
        experimental_vals = name_to_smiles.get_names(
            args['experimental_vals']
        )
    else:
        experimental_vals = ['' for _ in range(len(names))]
    smiles = name_to_smiles.get_smiles(names)

    # Aggregate names, SMILES, experimental values in tuples
    to_process = []
    for idx, name in enumerate(names):
        if smiles[idx] != '':
            to_process.append((name, smiles[idx], experimental_vals[idx]))

    # Write temporary .smiles file, obtain QSPR descriptors
    with open(args['smiles_file'], 'w') as smi_file:
        for mol in to_process:
            smi_file.write('{}\n'.format(mol[1]))
    molecules = smiles_to_qspr.get_descriptors(
        args['smiles_file'],
        clean_up=args['clean_up']
    )
    if args['clean_up']:
        remove(args['smiles_file'])

    # Create ECNet-formatted database
    print('Saving database: {}'.format(args['output_file']))
    fieldnames = sorted(list(molecules[0].keys()))
    fieldnames.remove('SMILES')
    rows = []
    type_row = ['DATAID', 'ASSIGNMENT', 'STRING', 'STRING', 'TARGET']
    type_row.extend(['INPUT' for _ in range(len(fieldnames))])
    title_row = ['DATAID', 'ASSIGNMENT', 'Compound Name', 'SMILES', 'Exp']
    title_row.extend(fieldnames)
    rows.append(type_row)
    rows.append(title_row)
    for idx, mol in enumerate(to_process):
        mol_row = [
            '{}'.format(args['id_prefix']) + '%04d' % (idx + 1),
            'L',
            mol[0],
            mol[1],
            mol[2]
        ]
        for fn in fieldnames:
            mol_row.append(molecules[idx][fn])
        rows.append(mol_row)

    with open(args['output_file'], 'w', encoding='utf-8') as output_file:
        wr = writer(output_file, quoting=QUOTE_ALL, lineterminator='\n')
        for row in rows:
            wr.writerow(row)


if __name__ == '__main__':

    main(parse_args())
