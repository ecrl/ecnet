#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/tools/plotting.py
# v.3.1.2
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains functions/classes for creating various plots
#

# stdlib. imports
from math import sqrt

# 3rd party imports
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


class ParityPlot:

    def __init__(self, title: str='Parity Plot',
                 x_label: str='Experimental Value',
                 y_label: str='Predicted Value', font: str='Times New Roman'):
        ''' ParityPlot: creates a plot of predicted values vs. experimental
        data relative to a 1:1 parity line

        Args:
            title (str): title of the plot
            x_label (str): x-axis label for the plot
            y_label (str): y-axis label for the plot
            font (str): font for the plot
        '''

        plt.rcParams['font.family'] = font
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        self._max_val = 0
        self._labels = None

    def add_series(self, x_vals, y_vals, name: str=None, color: str=None):
        ''' Adds data to the plot

        Args:
            x_vals (iter): x values for the series
            y_vals (iter): y values for the series (same length as x_vals)
            name (str): if not None, names the series and places legend
            color (str): if not None, uses specified color to denote series
        '''

        if len(x_vals) != len(y_vals):
            raise ValueError('Length of supplied X and Y values are not equal:'
                             ' {}, {}'.format(len(x_vals), len(y_vals)))
        plt.scatter(x_vals, y_vals, label=name, c=color)
        plt.legend(loc=2, edgecolor='w')
        x_max = max(x_vals)
        if x_max > self._max_val:
            self._max_val = x_max
        y_max = max(y_vals)
        if y_max > self._max_val:
            self._max_val = y_max

    def add_error_bars(self, error: float, label: str=None):
        ''' Adds error bars, +/- the error relative to the 1:1 parity line

        Args:
            error (int, float): error value
            label (str): if not None, adds the name/value to the legend
        '''

        self._add_parity_line(offset=error)
        self._add_parity_line(offset=(-1 * error))
        if label is not None:
            self._add_label(label, error)

    def show(self):
        ''' Shows the plot on-screen '''

        self._add_parity_line()
        plt.show()

    def save(self, filename: str):
        ''' Saves the plot to a file

        Args:
            filename (str): path to desired save location/file
        '''

        self._add_parity_line()
        plt.savefig(filename)

    def _add_parity_line(self, offset: float=0.0):
        ''' Adds a 1:1 parity line

        Args:
            offset (int, float): if not 0, adds a +/- offset relative to y=0
                when x=0 parity line
        '''

        if offset < 0:
            direction = -1
        else:
            direction = 1
        norm_offset = direction * sqrt(2 * offset * offset)
        plt.plot(
            [0, self._max_val],
            [0 + norm_offset, self._max_val + norm_offset],
            color='k',
            linestyle=':',
            zorder=0,
            linewidth=1
        )

    def _add_label(self, label: str, value: float):
        ''' Adds a label and value to the plot's legend

        Args:
            label (str): name of the label
            value (int, float): value of the label
        '''

        string = '{}: '.format(label) + '%.3f' % value
        if self._labels is None:
            self._labels = string
        else:
            self._labels += '\n' + string
        text_box = AnchoredText(self._labels, frameon=True, loc=4, pad=0.5)
        plt.setp(text_box.patch, facecolor='white', edgecolor='w')
        plt.gca().add_artist(text_box)
