from setuptools import setup, find_packages

setup(
    name='ecnet',
    version='3.2.2',
    description='UMass Lowell Energy and Combustion Research Laboratory Neural'
                ' Network Software',
    url='http://github.com/tjkessler/ecnet',
    author='Travis Kessler, Hernan Gelaf-Romer, Sanskriti Sharma',
    author_email='travis.j.kessler@gmail.com,'
                 ' Hernan_Gelafromer@student.uml.edu,'
                 ' Sanskriti_Sharma@student.uml.edu',
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
