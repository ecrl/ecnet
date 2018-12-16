#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/error_utils.py
# v.2.0.0
# Developed in 2018 by Travis Kessler <Travis_Kessler@student.uml.edu>
#
# Contains functions for error calculations
#

# 3rd party imports
from numpy import absolute, asarray, median, sqrt as nsqrt, sum as nsum


def calc_rmse(y_hat, y):

    try:
        return(nsqrt(((y_hat-y)**2).mean()))
    except:
        try:
            return(nsqrt(((asarray(y_hat)-asarray(y))**2).mean()))
        except:
            raise ValueError('Check input data format.')


def calc_mean_abs_error(y_hat, y):

    try:
        return(abs(y_hat-y).mean())
    except:
        try:
            return(abs(asarray(y_hat)-asarray(y)).mean())
        except:
            raise ValueError('Check input data format.')


def calc_med_abs_error(y_hat, y):

    try:
        return(median(absolute(y_hat-y)))
    except:
        try:
            return(median(absolute(asarray(y_hat)-asarray(y))))
        except:
            raise ValueError('Check input data format.')


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
    try:
        s_res = nsum((y_hat-y)**2)
        s_tot = nsum((y-y_mean)**2)
        return(1 - (s_res/s_tot))
    except:
        try:
            s_res = nsum((asarray(y_hat)-asarray(y))**2)
            s_tot = nsum((asarray(y)-y_mean)**2)
            return(1 - (s_res/s_tot))
        except:
            raise ValueError('Check input data format.')
