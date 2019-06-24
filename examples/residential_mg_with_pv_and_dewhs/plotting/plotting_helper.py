import matplotlib
import matplotlib.ticker
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pandas.plotting._converter import TimeSeries_TimedeltaFormatter as _TimedeltaFormatter

IDX = pd.IndexSlice

FIG_SAVE_PATH = os.path.abspath(
    r'C:\Users\chris\Documents\University_on_gdrive\Thesis\git_Thesis\Final_Thesis_Report\images\\')

lineWidthNorm = 1.5
lineWidthThick = 2
lineWidthVThick = 3
fontNormalSize = 11
legendFontSize = 10

plt.rc('font', family='serif')
plt.rc('font', size=fontNormalSize)
plt.rc('legend', fontsize=legendFontSize)
plt.rc('legend', framealpha=0.95)


##Helper Funcs##
A4_WIDTH = 210 / 25.4  # A4 page
A4_HEIGHT = 297 / 25.4
def get_fig_axes_A4(nrows=1, ncols=1, h_scale=1.0, v_scale=1.0, sharex='none', sharey='none', squeeze=True,
                    subplot_kw=None, gridspec_kw=None, **fig_kw):
    figsize = fig_kw.pop('figsize', [A4_WIDTH, A4_HEIGHT])
    figsize[0] *= h_scale
    figsize[1] *= v_scale
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             sharex=sharex, sharey=sharey, squeeze=squeeze,
                             subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,
                             figsize=figsize, **fig_kw)
    if squeeze and nrows * ncols == 1:
        return fig, np.array([axes])
    else:
        return fig, axes

_USE_TEX = True
def plot_with_tex(enable=True):
    global _USE_TEX
    if enable:
        plt.rc('text', usetex=True)

        preamble = [r'\usepackage{amsmath,amssymb,mathtools}',
                    r'\usepackage{underscore}',
                    r'\usepackage[per-mode=symbol]{siunitx}',
                    r'\usepackage{parskip}']

        plt.rc('text.latex', preamble=preamble)
        _USE_TEX = enable
    else:
        plt.rc('text', usetex=False)
        plt.rc('text.latex', preamble=[])
        _USE_TEX = enable

plot_with_tex(_USE_TEX)


def tex_s(string):
    global _USE_TEX
    if not _USE_TEX:
        string = string.replace(r"\,", ' ')
        string = string.replace(r"\;", ' ')
        string = string.replace(r"\text", '')
        string = string.replace(r"\si", '')
        string = string.replace("\\", '')
    return string


class TDelta_formater(_TimedeltaFormatter):
    @staticmethod
    def format_timedelta_ticks(x, pos, n_decimals):
        """
        Convert seconds to 'D days HH:MM:SS.F'
        """
        s, ns = divmod(x, 1e9)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        s = r'{:2d}:{:02d}'.format(int(h), int(m))
        return s
