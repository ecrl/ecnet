from setuptools import setup

setup(name = 'ecnet',
version = "1.2.6.dev1",
description = 'UMass Lowell Energy and Combustion Research Laboratory Neural Network Software',
url = 'http://github.com/tjkessler/ecnet',
author = 'Travis Kessler',
author_email = 'Travis_Kessler@student.uml.edu',
license = 'MIT',
packages = ['ecnet'],
install_requires = ["tensorflow","pyyaml"],
zip_safe = False)
