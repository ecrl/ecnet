from setuptools import setup

setup(
    name='ecnet',
    version='2.0.0',
    description='UMass Lowell Energy and Combustion Research Laboratory Neural Network Software',
    url='http://github.com/tjkessler/ecnet',
    author='Travis Kessler, Hernan Gelaf-Romer, Sanskriti Sharma',
    author_email='travis.j.kessler@gmail.com, Hernan_Gelafromer@student.uml.edu, Sanskriti_Sharma@student.uml.edu',
    license='MIT',
    packages=['ecnet'],
    install_requires=['pyyaml', 'ecabc', 'pygenetics', 'colorlogging', 'numpy', 'tensorflow'],
    zip_safe=False
)
