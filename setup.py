from setuptools import setup

setup(name = 'ecnet',
version = "1.3.0.dev1",
description = 'UMass Lowell Energy and Combustion Research Laboratory Neural Network Software',
url = 'http://github.com/tjkessler/ecnet',
author = 'Travis Kessler, Hernan Gelaf-Romer, Sanskriti Sharma',
author_email = 'Travis_Kessler@student.uml.edu, Hernan_Gelafromer@student.uml.edu, Sanskriti_Sharma@student.uml.edu',
license = 'MIT',
packages = ['ecnet'],
install_requires = ["tensorflow","pyyaml", "numpy"],
zip_safe = False)
