import numpy as np
import matplotlib.pyplot as plt
import pandas
import sklearn
import sklearn.metrics
from scipy.interpolate import interp1d
from scipy.stats.distributions import entropy
# import mpltools.style
# mpltools.style.use('ggplot')


def evaluate_performance(confidences, labels, loss_func, cumulative_costs,
                         max_budget, ylabel, filename=None, plot_figure=False,
                         plot_filename=None):
    """
    Evaluate the performance of a sequential classification by plotting it
    and returning the figure and the area under the curve.

    Additionally, write out the interpolated points and values to file.

    Parameters
    ----------
    confidences: list of (?, K) ndarrays of float
    labels: list of integers in [0, K]
    loss_func: callable
    cumulative_costs: list of (?,) ndarrays of float
    max_budget: float
    ylabel: string
    filename: string, optional
        To write out the interpolation points and values.
    plot_figure: bool, optional [False]
        If True, return figure containing plot in addition to auc value.
    plot_filename: string, optional
        If given, plot is written out to here.

    Returns
    -------
    auc: float
    fig: matplotlib figure if plot_figure==True or None
    """
    scores = []
    for c in xrange(len(confidences)):
        scores.append(loss_func(confidences[c], labels[c]))

    # find average of all interpolated performances
    num_interp_points = max_budget * 2
    interp_points = np.linspace(0, max_budget, num_interp_points)
    scores_s = []
    for c in xrange(len(scores)):
        if scores[c].shape[0] > 1:
            f = interp1d(cumulative_costs[c], scores[c],
                         bounds_error=False, fill_value=scores[c][-1])
            scores_s.append(f(interp_points))
        else:
            scores_s.append(np.repeat(scores[c], len(interp_points)))
    scores_s = np.vstack(scores_s)
    means = scores_s.mean(0)
    stds = scores_s.std(0)
    auc = round(sklearn.metrics.auc(interp_points, means) / max_budget, 3)
    final = round(means[-1], 3)

    if filename is not None:
        np.savez(filename, interp_points=interp_points, means=means, stds=stds)

    if plot_figure:
        # also plot points aggregated in a different way
        scores = np.hstack(scores)
        cumulative_costs = np.hstack(cumulative_costs)
        series = pandas.Series(scores, cumulative_costs).sort_index()
        g = series.groupby(series.index)

        # filter out points with little data
        count_fraction_threshold = 0.1
        counts = g.count().astype('f')
        count_fractions = counts / counts[0]
        ind = count_fractions > count_fraction_threshold
        count_fractions = count_fractions[ind]
        mean = g.mean()[ind]
        ind = mean.index.values

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cumulative_costs, scores, 'o', alpha=0.01)
        ax.plot(interp_points, means, '-k',
                label='auc/final: {:.3f}/{:.3f}'.format(auc, means[-1]))
        ax.scatter(ind, mean.values, 150 * count_fractions, 'r')
        try:
            ax.fill_between(
                interp_points, means - stds, means + stds, alpha=0.1)
        except:
            pass
        ax.set_xlabel('Cost')
        ax.set_ylabel(ylabel)
        ax.set_xlim([0, max_budget])
        ax.set_ylim([0, 1])
        ax.legend()

        if plot_filename is not None:
            plt.savefig(plot_filename)
        return auc, final, fig
    else:
        return auc, final, None


def compute_rewards(confidences, label, loss_func, cumulative_costs,
                    max_budget, gamma, rewards_mode, normalize_locally):
    """
    Compute the rewards of each action (corresponding to intervals between
    confidences) for a run on a single instance.

    Note that in cases where performance decreases, like [1, 0, 1],
    the rewards will not add up to the true area under the curve.
    This is fine, because:
    - intuitively, gains should be rewarded even if later actions mess
    with them
    - this should actually never happen with a good classifier, because
        information is never decreasing.

    Parameters
    ----------
    confidences: (a+1, K) ndarray of float
        a refers to the number of actions that were actually taken.

    label: int

    loss_func: callable
        For example, zero_one_loss or info_loss.

    cumulative_costs: (a+1,) ndarray of float
        Cumulative costs of the actions taken. First cost must be 0.
        Last cost must not exceed max_budget.

    max_budget: float
        The maximum feature cost budget.

    gamma: float
        The discount parameter.

    rewards_mode: string in ['auc', 'final']
        If 'final', only the last gain is considered, and costs are neglected.

    normalize_locally: bool
        If False, the sum of undiscounted rewards as computed corresponds
        exactly to the area under the of the (1 - losses) vs. costs curve,
        normalized by the total possible area under the budget.

        If True, the sum does not correspond to the normalized area, as the
        reward at each point is computed relative not to the total
        possible area
        but to the possible area from that point.

    Returns
    -------
    rewards: (a+1,) ndarray of float
    """
    if gamma is None:
        gamma = 0

    losses = loss_func(confidences, label)
    assert(losses.shape[0] == cumulative_costs.shape[0])
    gains = losses[:-1] - losses[1:]

    # Gains may be an empty array if no actions were actually taken.
    # In that case, return the 0 reward.
    if len(gains) == 0:
        return np.array([0])

    c = np.array(cumulative_costs)
    assert(c[0] == 0)
    assert(c[-1] <= max_budget)

    if rewards_mode == 'final':
        rewards = np.zeros_like(gains)
        rewards[-1] = gains[-1]
    elif rewards_mode == 'auc':
        # We are representing the area under the gain vs. cost curve
        # with rectangles from the midpoint of the action to the end
        # of the budget.
        midpoints = c[:-1] + (c[1:] - c[:-1]) / 2.
        dist_to_max_budget = max_budget - midpoints  # guaranteed non-negative

        areas = gains * dist_to_max_budget
        if normalize_locally:
            rewards = areas / (max_budget - c[:-1])
            rewards[rewards > 0] /= 1 - losses[rewards > 0]
            rewards[rewards < 0] /= losses[rewards < 0]
        else:
            rewards = areas / max_budget
    else:
        raise Exception("Unknown rewards_mode")

    if gamma > 0:
        rewards = discount_rewards(rewards, gamma)
    # last, untaken action gets 0 reward
    return np.hstack((rewards, 0))


def discount_rewards(rewards, gamma=1):
    if gamma == 0:
        return rewards
    x = np.array(rewards)
    d = np.zeros_like(x)
    for i in xrange(x.shape[0]):
        x[-1:-1 - i:-1] *= gamma
        d[i] = x[-1:-2 - i:-1].sum()
    return d[-1::-1]


def plot_trajectories(
        actions, rewards, ds, N=250, filename=None,
        figsize=(3, 1.85), fontsize=10):
    """
    Plot a parallel coordinates plot of actions taken.

    Parameters
    ----------
    actions: list of ndarrays of int

    rewards: list of ndarrays of float

    ds: tc.DataSource

    N: int, optional
        Number of (randomly sampled w/o replacement) trajectories to plot.

    filename: string, optional
        If given, plot is written out to here.

    figsize: tuple of float

    fontsize: int
        For the default figsize:
        - 10 is good for synthetic orthants,
        - 8 for scenes

    Returns
    -------
    fig: matplotlib figure
    """
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    plt.rc('text', usetex=True)

    plt.rc('axes', labelsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('legend', fontsize=fontsize)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111)
    num_actions = []
    sc = None
    for i in np.random.permutation(len(actions))[:N]:
        # don't plot the last action, as it is not actually taken
        a = actions[i].shape[0] - 1
        num_actions.append(a)
        if a > 0:
            ax.plot(range(a), actions[i][:-1], '-', color='gray', alpha=0.04)
            sc = ax.scatter(
                range(a), actions[i][:-1],
                c=rewards[i][:-1], cmap=plt.cm.RdBu_r, vmin=-1, vmax=1,
                s=33, alpha=0.4, edgecolor='gray', linewidths=.1,
            )

    max_num_actions = 0
    if sc is not None:
        max_num_actions = int(np.percentile(num_actions, 80))

    ax.set_xlim([0, max_num_actions])
    ax.set_xticks(range(max_num_actions + 1))
    ax.set_xlabel('Number in action sequence')

    ax.set_ylabel('Action')
    ax.set_ylim((0, len(ds.actions)))
    ax.set_yticks(range(len(ds.actions)))
    tex_safe_actions = [_.replace('_', '\_') for _ in ds.actions]
    ax.set_yticklabels(tex_safe_actions)

    # Pretty it up!
    for spine in ax.spines.itervalues():
        spine.set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # fig.tight_layout(pad=0.2)

    if filename:
        fig.savefig(filename, dpi=300)
        fig.savefig(filename + '.pdf')

    plt.rc('text', usetex=False)


def zero_one_loss(confidences, labels):
    """
    Return the 0-1 loss of the given confidences: for each row, the most
    confident prediction is compared for equality with the corresponding label.

    Parameters
    ----------
    confidences: (N, K) or (K,) ndarray of float

    labels: (N,) ndarray of int or int
        If single int is given, all rows are compared to it.

    Returns
    -------
    losses: (N,) ndarray of int
    """
    if confidences.ndim == 1:
        return (np.argmax(confidences) != labels).astype('int')
    else:
        return (np.argmax(confidences, axis=1) != labels).astype('int')


def info_loss(confidences, *args):
    """
    Return the loss defined by increase in entropy.

    Parameters
    ----------
    confidences: (N, K) or (K,) ndarray of float

    args: None
        To maintain compatibility with other loss funcions, we accept
        additional arguments and ignore them.

    Returns
    -------
    losses: (N,) ndarray of float or float
    """
    # guard against nan and -0
    if confidences.ndim == 1:
        scores = entropy(confidences) / np.log(confidences.shape[0])
        if np.isnan(scores):
            return 1
    else:
        scores = entropy(confidences.T) / np.log(confidences.shape[1])
        scores[np.isnan(scores)] = 1
    return np.maximum(0, scores)
