[![UML Energy & Combustion Research Laboratory](http://faculty.uml.edu/Hunter_Mack/uploads/9/7/1/3/97138798/1481826668_2.png)](http://faculty.uml.edu/Hunter_Mack/)

# ECNet: scalable, retrainable and deployable machine learning projects for fuel property prediction

[![GitHub version](https://badge.fury.io/gh/tjkessler%2FECNet.svg)](https://badge.fury.io/gh/tjkessler%2FECNet)
[![PyPI version](https://badge.fury.io/py/ecnet.svg)](https://badge.fury.io/py/ecnet)
[![status](http://joss.theoj.org/papers/f556afbc97e18e1c1294d98e0f7ff99f/status.svg)](http://joss.theoj.org/papers/f556afbc97e18e1c1294d98e0f7ff99f)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/TJKessler/ECNet/master/LICENSE.txt)
[![Documentation Status](https://readthedocs.org/projects/ecnet/badge/?version=latest)](https://ecnet.readthedocs.io/en/latest/?badge=latest)
	
**ECNet** is an open source Python package for creating scalable, retrainable and deployable machine learning projects with a focus on fuel property prediction. An ECNet __project__ is considered a collection of __pools__, where each pool contains a neural network that has been selected from a group of __candidate__ neural networks. Candidates are selected to represent pools based on their ability to optimize certain learning criteria (for example, performing optimially on unseen data). Each pool contributes a prediction derived from input data, and these predictions are averaged to calculate the project's final prediction. Using multiple pools allows a project to learn from a variety of learning and validation sets, which can reduce the project's prediction error. Projects can be saved and reused at a later time allowing additional training and deployable predictive models. 

[T. Sennott et al.](https://doi.org/10.1115/ICEF2013-19185) have shown that neural networks can be applied to cetane number prediction with relatively little error. ECNet provides scientists an open source tool for predicting key fuel properties of potential next-generation biofuels, reducing the need for costly fuel synthesis and experimentation.

<p align="center">
  <img align="center" src="docs/img/workflow_diagram.png" width="50%" height="50%">
</p>

Using ECNet, [T. Kessler et al.](https://doi.org/10.1016/j.fuel.2017.06.015) have increased the generalizability of neural networks to predict the cetane number for a variety of molecular classes represented in our [cetane number database](https://github.com/TJKessler/ECNet/tree/master/databases), and have increased the accuracy of neural networks for predicting the cetane number of underrepresented molecular classes through targeted database expansion.

Future plans for ECNet include:
- distributed candidate training for GPU's
- a graphical user interface
- implementing neural network neuron diagnostics - maybe it's not a black box after all 🤔

# Installation and Usage

Please refer to our [documentation page](https://ecnet.readthedocs.io/en/latest/) for installation instructions, a quick-start guide, and how-to's for various tools ECNet provides.

# Contributing, Reporting Issues and Other Support:

To contribute to ECNet, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (travis.j.kessler@gmail.com), Hernan Gelaf-Romer (hernan_gelafromer@student.uml.edu) and/or John Hunter Mack (Hunter_Mack@uml.edu).
