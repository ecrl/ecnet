from csv import DictReader
from typing import List

_properties = [
    'cetane_number',
    'research_octane_number',
    'motor_octane_number',
    'octane_sensitivity',
    'ysi_unified',
    'autoignition_temp',
    'boiling_point',
    'flash_point',
    'heat_of_vaporization',
    'melting_point',
    'kinematic_viscosity',
    'pour_point'
]


def get_descriptors(cas: List[str], db_loc: str = 'descriptors_master.csv') -> List[dict]:
    r"""
    Obtains all descriptors for a list of compounds (list of their CAS numbers)

    Args:
        cas list[str]: list of CAS numbers, one per compound
        db_loc (str, optional): location od CombustDB-exported descriptor database; default
            location is ./descriptors_master.csv

    Returns:
        list[dict]: each list element (dict) is a compound's descriptors from alvaDesc and PaDEL-
            Descriptor, plus CAS number
    """

    with open(db_loc, 'r') as csv_file:
        reader = DictReader(csv_file)
        compounds = [c for c in reader]
    csv_file.close()

    keys = list(compounds[0].keys())
    keys.remove('cas')

    to_return = []
    for cs in cas:
        found = False
        for comp in compounds:
            if cs == comp['cas']:
                for key in keys:
                    if comp[key] == 'na' or comp[key] == '' or comp[key] == '-':
                        comp[key] = 0.0
                    else:
                        comp[key] = float(comp[key])
                to_return.append(comp)
                found = True
                break
        if not found:
            raise IndexError('Could not find descriptors for CAS `{}`'.format(cs))
    return to_return


def get_compounds(prop: str, db_loc: str = 'properties_master.csv') -> List[dict]:
    r"""
    Returns all compounds that have experimental data for user-specified property

    Args:
        prop (str): name of property; available properties are:
            `cetane_number`,
            `research_octane_number`,
            `motor_octane_number`,
            `octane_sensitivity`,
            `ysi_unified`,
            `autoignition_temp`,
            `boiling_point`,
            `flash_point`,
            `heat_of_vaporization`,
            `melting_point`,
            `kinematic_viscosity`,
            `pour_point`
        db_loc (str, optional): location of CombustDB-exported property database; default location
            is ./properties_master.csv

    Returns:
        list[dict]: each list element (dict) is a compound which has experimental data for user-
            specified property
    """

    if prop not in _properties:
        raise ValueError('{} not found in available properties: {}'.format(prop, _properties))

    with open(db_loc, 'r') as csv_file:
        reader = DictReader(csv_file)
        compounds = [c for c in reader]
    csv_file.close()

    to_return = []
    for comp in compounds:
        if comp['properties.{}.value'.format(prop)] != '-':
            to_return.append(comp)
        else:
            continue
    return to_return


# Example: get property values, descriptors for database subset w/ cetane number data
if __name__ == '__main__':

    # Obtain compounds with experimental cetane number data
    comps = get_compounds('cetane_number')

    # Obtain descriptors for compounds with experimental cetane number data
    desc = get_descriptors([c['cas'] for c in comps])

    # Get SMILES strings, cetane number values, descriptors for downstream processing
    smiles = [c['canonical_smiles'] for c in comps]
    cn = [float(c['properties.cetane_number.value']) for c in comps]
    keys = list(desc[0].keys())
    keys.remove('cas')
    desc = [[d[k] for k in keys] for d in desc]

    print('Number of SMILES strings: {}'.format(len(smiles)))
    print('Number of cetane number values: {}'.format(len(cn)))
    print('Shape of descriptors matrix: ({}, {})'.format(len(desc), len(desc[0])))
