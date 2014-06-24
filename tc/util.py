import matplotlib.pyplot as plt
import numpy as np
import os
import errno
import cPickle as pickle
import time


def load_pickle(filename):
    with open(filename) as f:
        return pickle.load(f)


def dump_pickle(obj, filename):
    with open(filename, 'w') as f:
        pickle.dump(obj, f, protocol=2)


class Timer(object):
    """
    Convenience class to time bits of code.
    """
    def __init__(self):
        self.times = {}

    def tic(self, name):
        self.times[name] = [time.time(), -1]

    def qtoc(self, name):
        return self.toc(name, quiet=True)

    def toc(self, name, quiet=False):
        if name in self.times:
            t = time.time()
            self.times[name][1] = t
            te = round(t - self.times[name][0], 3)
            if not quiet:
                print('Timer.{}: {} s'.format(name, te))
            return te

    def report(self):
        return dict([(name, round(t[1] - t[0], 3)) for name, t in self.times.items()])


def slice_array(array, bounds, name):
    """
    Return slice of the vector corresponding to the given name.
    If the array is 2-D, slices columns.

    Parameters
    ----------
    array: (M, N) ndarray

    bounds: dict
        Dictionary of name to slicing tuple.

    name: string
        Key into the bounds dict.
    """
    if array.ndim == 1:
        return array[slice(*bounds[name])]
    elif array.ndim == 2:
        return array[:, slice(*bounds[name])]
    else:
        raise("More than 2 dimensions not supported.")


def plot_weights(weights, xlabel=None, xticks=None, ylabel=None, yticks=None, filename=None):
    """
    Return matplotlib figure with coefficient weights.

    Parameters
    ----------
    weights: (M, N) ndarray
    
    xlabel: string, optional

    xticks: list of strings, optional
    
    ylabel: string, optional

    yticks: list of strings, optional

    filename: string, optional

    Returns
    -------
    fig: matplotlib figure
    """
    fig = plt.figure()
    abs_max_weight = np.max(np.abs(weights))
    plt.matshow(
        weights, cmap=plt.cm.RdBu_r,
        vmin=-abs_max_weight, vmax=abs_max_weight)
    plt.colorbar()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if xticks is not None:
        plt.xticks(np.arange(len(xticks)), xticks)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if yticks is not None:
        plt.yticks(np.arange(len(yticks)), yticks)
    if filename:
        plt.savefig(filename)
    return fig

    
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
