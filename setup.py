from setuptools import find_packages, setup

setup(
    name='ecnet',
    version='4.1.1',
    description='Fuel property prediction using QSPR descriptors',
    url='https://github.com/ecrl/ecnet',
    author='Travis Kessler',
    author_email='Travis_Kessler@student.uml.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch==2.0.0',
        'scikit-learn==1.2.2',
        'padelpy==0.1.13',
        'alvadescpy==0.1.2',
        'ecabc==3.0.0'
    ],
    package_data={
        'ecnet': [
            'datasets/data/*'
        ]
    },
    zip_safe=False
)
