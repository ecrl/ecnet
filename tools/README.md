# Creating an ECNet-formatted database from molecule names, target values

### Prerequisites

To create an ECNet-formatted database from molecule names and target values, you must have the following packages/programs installed:
- [PubChemPy](https://github.com/mcs07/PubChemPy) Python package
- [Open Babel](http://openbabel.org/wiki/Main_Page) software
- [Java JRE](https://www.oracle.com/technetwork/java/javase/downloads/jre8-downloads-2133155.html) version 6 and above

Additionally, a distributable version of [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) is required for QSPR descriptor generation (we have plans for adding a wrapper for Dragon software too!) - we have included a distributed version of PaDEL-Descriptor in this directory.

All source files in this directory are required to create a database, and must exist next to each other in your current working directory:
- **create_ecnet_db.py** (aggregates SMILES, QSPR descriptors for database creation)
- **name_to_smiles.py** (uses PubChemPy to obtain SMILES strings for supplied molecule names)
- **smiles_to_qspr.py** (generates QSPR descriptors from SMILES strings using Open Babel and PaDEL-Descriptor)

You also need a .txt file populated with molecule names, one per line:
```
Acetaldehyde
Acetaldehyde dimethyl acetal
Acetic acid
Acetic anhydride
Acetol
Acetone
Acetonitrile
Acetonylacetone
```

And another .txt file populated with target (experimental) values (as an example, here are the molecules' boiling points in degrees Fahrenheit):
```
70
147
244
284
295
133
180
376
```
Target values are optional; if no target values are supplied, the resulting database will not have its TARGET column populated. The supplied targets file must be of equal length to the supplied molecules file.

### Creating a database

Run create_ecnet_db.py from the command line with:
```console
foo@bar:~$ python create_ecnet_db.py molecules.txt my_database.csv --experimental_vals targets.txt
```
Where molecules.txt is the path to your molecules file, my_database.csv is the name of the database you are creating, and targets.txt are your target values.

### Additional options

create_ecnet_db.py assumes your version of PaDEL-Descriptor is adjacent to the script (located at .\PaDEL-Descriptor\PaDEL-Descriptor.jar). If your version of PaDEL-Descriptor is located elsewhere, you can supply the "--padel_path" argument:
```console
foo@bar:~$ python create_ecnet_db.py molecules.txt my_database.csv --experimental_vals targets.txt --padel_path \path\to\PaDEL-Descriptor.jar
```

Your database's DATAID column (essentially Bates numbers for each molecule) will increment starting at 0001:

| DATAID 	|
|--------	|
| DATAID 	|
| 0001   	|
| 0002   	|
| 0003   	|

To include a prefix for your molecule ID's, supply the "--id-prefix" argument:

```console
foo@bar:~$ python create_ecnet_db.py molecules.txt my_database.csv --experimental_vals targets.txt --id_prefix MOL
```

| DATAID 	    |
|-----------	|
| DATAID     	|
| MOL0001   	|
| MOL0002   	|
| MOL0003   	|
