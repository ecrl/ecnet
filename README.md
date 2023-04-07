[![UML Energy & Combustion Research Laboratory](https://sites.uml.edu/hunter-mack/files/2021/11/ECRL_final.png)](http://faculty.uml.edu/Hunter_Mack/)

# ECNet: machine learning models for fuel property prediction

[![GitHub version](https://badge.fury.io/gh/ecrl%2FECNet.svg)](https://badge.fury.io/gh/ecrl%2FECNet)
[![PyPI version](https://badge.fury.io/py/ecnet.svg)](https://badge.fury.io/py/ecnet)
[![status](http://joss.theoj.org/papers/f556afbc97e18e1c1294d98e0f7ff99f/status.svg)](http://joss.theoj.org/papers/f556afbc97e18e1c1294d98e0f7ff99f)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/ECRL/ECNet/master/LICENSE.txt)
[![Documentation Status](https://readthedocs.org/projects/ecnet/badge/?version=latest)](https://ecnet.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://dev.azure.com/uml-ecrl/package-management/_apis/build/status/ECRL.ECNet?branchName=master)](https://dev.azure.com/uml-ecrl/package-management/_build/latest?definitionId=1&branchName=master)
	
**ECNet** is an open source Python package for creating machine learning models to predict fuel properties. ECNet comes bundled with a variety of fuel property datasets, including cetane number, yield sooting index, and research/motor octane number. ECNet was built using the [PyTorch](https://pytorch.org/) library, allowing easy implementation of our models in your existing ML pipelines.

ECNet leverages [QSPR descriptors](https://en.wikipedia.org/wiki/Quantitative_structure%E2%80%93activity_relationship) for use as input variables, specifically [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) and [alvaDesc](https://www.alvascience.com/alvadesc/). Using alvaDesc requires a valid license.

Future plans for ECNet include:
- Implementating RDKit to train using molecular fingerprints
- Leveraging additional QSPR-generation software packages (e.g. [Mordred](https://github.com/mordred-descriptor/mordred))
- A graphical user interface

# Installation and Usage

Please refer to our [documentation page](https://ecnet.readthedocs.io/en/latest/) for installation instructions and full API documentation. You can also view some [example scripts](https://github.com/ECRL/ECNet/tree/master/examples) we put together.

# Contributing, Reporting Issues, and Other Support:

To contribute to ECNet, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (Travis_Kessler@student.uml.edu) and/or John Hunter Mack (Hunter_Mack@uml.edu).
