#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/error_utils.py
# v.2.1.0
# Developed in 2019 by Travis Kessler <Travis_Kessler@student.uml.edu>
#
# Contains functions for error calculations
#

# 3rd party imports
from numpy import absolute, asarray, float64, isinf, isnan, median,\
    nan_to_num, square, sqrt as nsqrt, sum as nsum


def calc_rmse(y_hat, y):

    rmse = nsqrt((square(_get_diff(y_hat, y)).mean()))
    if isnan(rmse) or isinf(rmse):
        return 99
    return rmse


def calc_mean_abs_error(y_hat, y):

    mae = absolute(_get_diff(y_hat, y)).mean()
    if isnan(mae) or isinf(mae):
        return 99
    return mae


def calc_med_abs_error(y_hat, y):

    medae = median(absolute(_get_diff(y_hat, y)))
    if isnan(medae) or isinf(medae):
        return 99
    return medae


def calc_r2(y_hat, y):

    try:
        y_mean = y.mean()
    except:
        try:
            y_form = []
            for i in range(len(y)):
                y_form.append(y[i][0])
            y_mean = sum(y_form)/len(y_form)
        except:
            raise ValueError('Check input data format.')

    s_res = nsum(_get_diff(y_hat, y)**2)
    s_tot = nsum(_get_diff(y, y_mean)**2)
    return(1 - (s_res / s_tot))


def _get_diff(y_hat, y):

    return asarray(y_hat, dtype=float64) - asarray(y, dtype=float64)
