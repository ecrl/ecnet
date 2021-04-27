# ECNet Documentation

[![UML Energy & Combustion Research Laboratory](http://faculty.uml.edu/Hunter_Mack/uploads/9/7/1/3/97138798/1481826668_2.png)](http://faculty.uml.edu/Hunter_Mack/)

[![GitHub version](https://badge.fury.io/gh/ecrl%2FECNet.svg)](https://badge.fury.io/gh/ecrl%2FECNet)
[![PyPI version](https://badge.fury.io/py/ecnet.svg)](https://badge.fury.io/py/ecnet)
[![status](http://joss.theoj.org/papers/f556afbc97e18e1c1294d98e0f7ff99f/status.svg)](http://joss.theoj.org/papers/f556afbc97e18e1c1294d98e0f7ff99f)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/ECRL/ECNet/master/LICENSE.txt)
[![Documentation Status](https://readthedocs.org/projects/ecnet/badge/?version=latest)](https://ecnet.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://dev.azure.com/uml-ecrl/package-management/_apis/build/status/ECRL.ECNet?branchName=master)](https://dev.azure.com/uml-ecrl/package-management/_build/latest?definitionId=1&branchName=master)

## Installation

### Installation via pip

    pip install ecnet

Additional dependencies (torch, sklearn, padelpy, alvadescpy, ecabc) will be installed during ECNet's installation process. If you have any trouble with these dependencies (or want to compile them yourself, e.g. PyTorch with GPU support), consider installing them from source before installing ECNet.

### Upgrading via pip

    pip install --upgrade ecnet

### Installation from source

    git clone https://github.com/ecrl/ecnet
    cd ecnet
    python setup.py install
