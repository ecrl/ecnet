from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from math import sqrt


class ParityPlot:

    def __init__(self, title='Parity Plot', x_label='Experimental Value',
                 y_label='Predicted Value', font='Times New Roman'):

        plt.rcParams['font.family'] = font
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        self._max_val = 0
        self._labels = None

    def add_series(self, x_vals, y_vals, name=None, color=None):

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

    def add_error_bars(self, error, label=None):

        self._add_parity_line(offset=error)
        self._add_parity_line(offset=(-1 * error))
        if label is not None:
            self._add_label(label, error)

    def show(self):

        self._add_parity_line()
        plt.show()

    def save(self, filename):

        self._add_parity_line()
        plt.savefig(filename)

    def _add_parity_line(self, offset=0):

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

    def _add_label(self, label, value):

        string = '{}: '.format(label) + '%.3f' % value
        if self._labels is None:
            self._labels = string
        else:
            self._labels += '\n' + string
        text_box = AnchoredText(self._labels, frameon=True, loc=4, pad=0.5)
        plt.setp(text_box.patch, facecolor='white', edgecolor='w')
        plt.gca().add_artist(text_box)
