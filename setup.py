from setuptools import setup, find_packages

setup(
    name='ecnet',
    version='3.2.3',
    description='UMass Lowell Energy and Combustion Research Laboratory Neural'
                ' Network Software',
    url='https://github.com/ecrl/ecnet',
    author='Travis Kessler, John Hunter Mack',
    author_email='Travis_Kessler@student.uml.edu,'
                 ' Hunter_Mack@uml.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'alvadescpy==0.1.0',
        'colorlogging==0.3.5',
        'ecabc==2.2.3',
        'keras==2.2.4',
        'matplotlib==3.1.0',
        'numpy==1.16.4',
        'padelpy==0.1.6',
        'pyyaml==5.1.1',
        'scikit-learn==0.21.2',
        'tensorflow==1.13.1'
    ],
    zip_safe=False
)
