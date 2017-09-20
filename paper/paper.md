---
title: 'ECNet: Large scale machine learning projects for fuel property prediction'
tags:
  - machine learning
  - artificial neural networks
  - fuel property prediction
  - cetane number
  - QSPR
authors:
 - name: Travis Kessler
   orcid: 0000-0002-7363-4050
   affiliation: 1
 - name: John Hunter Mack
   orcid: N/A
   affiliation: 1
affiliations:
 - name: UMass Lowell Energy and Combustion Research Laboratory
   index: 1
date: 4 September 2017
bibliography: paper.bib
nocite: |
  @ecnet_github, @ecnet_pypi, @uml_ecrl
---

# Summary

ECNet is an open source Python package for creating large scale machine learning projects with a focus on fuel property prediction. ECNet can predict a variety of fuel properties including cetane number, octane number and yield sooting index using quantitative structure-property relationship (QSPR) input parameters.

A project is considered a collection of builds, and each build is a collection of nodes. Nodes are averaged to obtain a final predicted value for the build. For each node in a build, multiple neural networks are constructed and the best performing neural network is used as that node's predictor. Using multiple nodes allows a build to learn from multiple learning and validation sets, reducing the build’s error.

T. Sennott [@asme_2013] et al. have shown that neural networks can be applied to cetane number prediction with relatively little error. ECNet provides scientists an open source tool for predicting key fuel properties of potential next-generation biofuels, reducing the need for costly fuel synthesis and experimentation.

Using ECNet, T. Kessler et al. have increased the generalizability of neural networks to predict the cetane numbers for molecules from a variety of molecular classes, and have increased the accuracy of neural networks for predicting the cetane numbers for molecules from underrepresented molecular classes through targeted database expansion.

Project(s) using ECNet:

Artificial neural network based predictions of cetane number for furanic biofuel additives [@fuel_2017]

Predicting the cetane number of furanic biofuel candidates using an improved artificial neural network based on molecular structure [@asme_2016]


# References
