import cPickle as cP
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import numpy
from glob import glob

k_colors = ["r", "b", "y", "m", "c", "g", "#FFA500", "k", "#77FF55"];
k_markers = "o*dxs^vDh";

def cmc_figure(methods, fname=None, keys=None, rank_lim=None):
    legfont = mpl.font_manager.FontProperties(family="monospace")
    # mlabdefaults()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    r1 = []
    result = []
    if keys is None:
        keys = methods.keys()
    for i, m in enumerate(keys):
        y = methods[m]
        x = np.arange(y.size)
        r1.append(y[0])
        ax.plot(x, y,
                color=k_colors[i],
                linestyle="-",
                marker=k_markers[i],
                markevery=5)
        result.append(y)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Identification Rate")
    if rank_lim is None:
        ax.set_xlim(0, 100)
    else:
        ax.set_xlim(0, rank_lim)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    r1 = ['%0.2f' % (r * 100) for r in r1]
    digitlen = len("10.10")
    total_len = len("Euclidean 10.10")

    r1 = ['%s%s(%s%s%%)' % (leg,
                            ''.join([' '] * (total_len - digitlen - len(leg))),
          ''.join([' '] * (digitlen - len(r))),
        r) for r, leg in zip(r1, keys)]
    ax.legend(r1, prop=legfont, loc=4)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    if fname is None:
        plt.show()
    else:
        fig.savefig(fname)
    plt.clf()
    return result

if __name__ == "__main__":
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['savefig.dpi'] = 150
    mpl.rcParams['font.size'] = 22.
    mpl.rcParams['font.family'] = "Times New Roman"
    mpl.rcParams['legend.fontsize'] = "small"
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['lines.markersize'] = 10
    mpl.rcParams['figure.figsize'] = 8, 5.6
    mpl.rcParams['legend.labelspacing'] = 0.1
    mpl.rcParams['legend.borderpad'] = 0.1
    mpl.rcParams['legend.borderaxespad'] = 0.2
    mpl.rcParams['font.monospace'] = "Courier New"

    keys = ['FPNN', "Euclidean", 'ITML', 'LMNN', 'RANK', 'LDM', "SDALF", 'eSDC', 'KISSME']

    mat = loadmat('cmc_figures_data/cuhk03_detected.mat')
    methods = {"ITML": mat['itml'],
               "FPNN": mat['fpnn'],
               "LDM": mat['ldm'],
               "LMNN": mat['lmnn'],
               "KISSME": mat['kissme'],
               "Euclidean": mat['euclidean'],
               "eSDC" : mat['esdc'],
               'RANK': mat['rank'],
               "SDALF": mat['sdalf']}
    cmc_figure(methods, "cmc_cuhk03_detected.eps", keys=keys)