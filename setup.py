from setuptools import setup, find_packages
from sys import argv

install_requires = [
    'alvadescpy==0.1.0',
    'colorlogging==0.3.5',
    'ecabc==3.0.0',
    'matplotlib==3.1.2',
    'numpy==1.16.4',
    'padelpy==0.1.6',
    'pyyaml==5.1.1',
    'scikit-learn==0.21.2'
]

if '--omit_tf' in argv:
    argv.remove('--omit_tf')
else:
    install_requires.append('tensorflow==2.0.0')

setup(
    name='ecnet',
    version='3.3.2',
    description='UMass Lowell Energy and Combustion Research Laboratory Neural'
                ' Network Software',
    url='https://github.com/ecrl/ecnet',
    author='Travis Kessler, John Hunter Mack',
    author_email='Travis_Kessler@student.uml.edu,'
                 ' Hunter_Mack@uml.edu',
    license='MIT',
    packages=find_packages(),
    python_requires='~=3.7',
    install_requires=install_requires,
    zip_safe=False
)
