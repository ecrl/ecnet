#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/utils/error_utils.py
# v.3.1.1
# Developed in 2019 by Travis Kessler <Travis_Kessler@student.uml.edu>
#
# Contains functions for error calculations
#

# 3rd party imports
from numpy import absolute, asarray, array, float64, median, square,\
    sqrt as nsqrt
from sklearn.metrics import r2_score


def calc_rmse(y_hat: array, y: array) -> float:

    return nsqrt((square(_get_diff(y_hat, y)).mean()))


def calc_mean_abs_error(y_hat: array, y: array) -> float:

    return absolute(_get_diff(y_hat, y)).mean()


def calc_med_abs_error(y_hat: array, y: array) -> float:

    return median(absolute(_get_diff(y_hat, y)))


def calc_r2(y_hat: array, y: array) -> float:

    return r2_score(y, y_hat)


def _get_diff(y_hat: array, y: array) -> array:

    return asarray(y_hat, dtype=float64) - asarray(y, dtype=float64)
