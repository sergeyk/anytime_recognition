import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import time
import networkx as nx


def eval_reward(preds, labels, rewards, graph, fast=False):
    """
    Parameters
    ----------
    fast : boolean, optional
        If True, then do not compute height_portion and height_acc; return None
        for them.
    """
    n = preds.shape[0]
    m = len(graph['nodes'])
    heights = graph['heights'] + 1  # to make the height stats calculations work
    k = (heights == 1).sum()

    pred_vec = np.zeros((n, m))
    pred_vec[np.arange(n), preds] = 1
    gt_vec = np.zeros((n, k))
    gt_vec[np.arange(n), labels] = 1
    gt_vec_full = np.dot(gt_vec, graph['leaf_membership'])

    correct_vec = gt_vec_full * pred_vec
    acc = np.sum(correct_vec) / n
    reward = np.sum(correct_vec * rewards) / n

    height_acc = height_portion = None
    if not fast:
        height_pred_vec = pred_vec * heights
        height_counts = 1. * np.histogram(height_pred_vec, 1 + np.arange(np.max(heights) + 1))[0]
        height_correct_vec = correct_vec * heights
        height_goods = 1. * np.histogram(height_correct_vec, 1 + np.arange(np.max(heights) + 1))[0]
        height_acc = height_goods / height_counts
        height_portion = height_counts / n

    return reward, acc, height_portion, height_acc


def binofit_scalar(x, n, alpha):
    '''Parameter estimates and confidence intervals for binomial data.
    (p,ci) = binofit(x,N,alpha)

    Source: Matlab's binofit.m
    Reference:
      [1]  Johnson, Norman L., Kotz, Samuel, & Kemp, Adrienne W.,
      "Univariate Discrete Distributions, Second Edition", Wiley
      1992 p. 124-130.
    http://books.google.com/books?id=JchiadWLnykC&printsec=frontcover&dq=Univariate+Discrete+Distributions#PPA131,M1

    Re-written by Santiago Jaramillo - 2008.10.06
    https://github.com/sjara/extracellpy/blob/master/extrastats.py
    '''

    if n < 1:
        Psuccess = np.nan
        ConfIntervals = (np.nan, np.nan)
    else:
        Psuccess = float(x)/n
        nu1 = 2 * x
        nu2 = 2 * (n - x + 1);
        F = scipy.stats.f.ppf(alpha / 2, nu1, nu2)
        lb = (nu1 * F) / (nu2 + nu1 * F)
        if x == 0:
            lb = 0
        nu1 = 2 * (x + 1)
        nu2 = 2 * (n - x)
        F = scipy.stats.f.ppf(1 - alpha / 2, nu1, nu2)
        ub = (nu1 * F) / (nu2 + nu1 * F)
        if x == n:
            ub = 1
        ConfIntervals = (lb, ub);
    return (Psuccess, ConfIntervals)


def darts_bisection(leaf_probs, acc_guarantees, labels, graph, num_bs_iters, confidence):
    t = time.time()
    labels = labels.flatten()
    n = leaf_probs.shape[0]
    lambdas = []
    for ag in acc_guarantees:
        eps = 1 - ag
        desired_alpha = (1 - confidence) * 2.
        min_lambda = 0
        max_lambda = ((1 - eps) * np.max(graph['rewards']) - np.min(graph['rewards'])) / eps
        for j in xrange(num_bs_iters):
            cur_lambda = (min_lambda + max_lambda) / 2.
            used_rewards = graph['rewards'] + cur_lambda
            preds = (np.dot(leaf_probs, graph['leaf_membership']) * used_rewards).argmax(1)
            reward, accuracy, _, _ = eval_reward(preds, labels, used_rewards, graph, fast=True)
            _, acc_bounds = binofit_scalar(accuracy * n, n, desired_alpha)
            if acc_bounds[0] > ag:
                max_lambda = cur_lambda
            else:
                min_lambda = cur_lambda
        lambdas.append(max_lambda)
    print('darts_bisection took {:.3f} s'.format(time.time() - t))
    return np.array(lambdas)


def darts_eval(leaf_probs, labels, lambdas, graph):
    assert(leaf_probs.shape[0] == labels.shape[0])

    t = time.time()
    labels = labels.flatten()
    normed_rewards = graph['rewards'] / np.max(graph['rewards'])

    num_heights = len(np.unique(graph['heights']))
    rewards = np.zeros_like(lambdas)
    accuracies = np.zeros_like(lambdas)
    height_portions = np.zeros((num_heights, lambdas.shape[0]))
    height_accs = np.zeros((num_heights, lambdas.shape[0]))

    for i, lam in enumerate(lambdas):
        used_rewards = graph['rewards'] + lam
        preds = (np.dot(leaf_probs, graph['leaf_membership']) * used_rewards).argmax(1)
        reward, acc, height_portion, height_acc = eval_reward(
            preds, labels, normed_rewards, graph, fast=False)
        rewards[i] = reward
        accuracies[i] = acc
        height_portions[:, i] = height_portion
        height_accs[:, i] = height_acc
    print('darts_eval took {:.3f} s'.format(time.time() - t))
    return rewards, accuracies, height_portions, height_accs


def plot_accuracy_vs_specificity(rewards, accuracies, labels=None):
    if rewards.ndim == 1:
        rewards = np.atleast_2d(rewards)
        accuracies = np.atleast_2d(accuracies)

    _labels = labels
    if labels is None:
        _labels = [''] * rewards.shape[0]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax2 = ax.twiny()

    cmap = plt.get_cmap('YlOrBr')
    colors = [cmap(i) for i in np.linspace(.3, 1, rewards.shape[0])]
    ax2.plot(rewards[0], accuracies[0], alpha=0)  # just to get the axis up
    if rewards.shape[0] == 1:
        ax.plot(rewards[0], accuracies[0], 's-', color='k', label=_labels[0])
    else:
        for i in xrange(rewards.shape[0]):
            ax.plot(
                rewards[i], accuracies[i], 's-',
                color=colors[i], label=_labels[i])

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Normalized information gain')
    ax.set_xlim([0, .5])
    ax2.set_xlim([0, .5])
    ax.set_ylim([0, 1.1])
    ax2.set_ylim([0, 1.1])
    num_classes = np.round(100 * 2 ** (-ax.get_xticks() * np.log2(57)), 0).astype('int')
    ax2.set_xticklabels(num_classes)
    ax2.set_xlabel('% Uncertain classes')
    if labels is not None:
        ax.legend()
        ax.set_xlim([0, .6])
        ax2.set_xlim([0, .6])
    return fig


def meta_evaluate(graph, budgets, rewards, accuracies, height_portions):
    plot_accuracy_vs_specificity(
        rewards, accuracies, labels=['{:.1f}'.format(b) for b in budgets])

    cmap = plt.get_cmap('Accent')
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)

    # aucs
    aucs = [-np.trapz(accuracies[i], rewards[i]) for i in range(accuracies.shape[0])]
    ax.plot(budgets, aucs, 's-', color=cmap(0.3),
            label='Area under Acc vs. Specificity: {:.3f}'.format(np.trapz(aucs), budgets))

    # mean accuracies
    # TODO: this isn't right: this is not the guarantee
    ax.plot(budgets, accuracies.min(1), 's-', color=cmap(0.6),
            label='Mean Accuracy: {:.3f}'.format(np.trapz(accuracies.mean(1), budgets)))

    # rewards at acc=0.9
    max_rewards = np.zeros(rewards.shape[0])
    for i in xrange(rewards.shape[0]):
        ind = np.where(accuracies[i] >= .9)[0]
        if len(ind) > 0:
            max_rewards[i] = rewards[i][ind[0]]
    ax.plot(budgets, max_rewards, 's-', color=cmap(0.8),
            label='Specificity at Acc=0.9: {:.3f}'.format(np.trapz(max_rewards, budgets)))
    ax.set_xlabel('Budget')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, .8])
    plt.legend()

    plot_height_portions(graph, budgets, height_portions)
    return None


def plot_height_portions(graph, budgets, height_portions):
    """
    For a few budget points, plot the average portion of predictions (across all accuracies)
    by filling in the ILSVRC65 nodes.
    """
    g = graph['g']
    nodes = graph['nodes']
    pos = nx.pygraphviz_layout(g, prog='twopi')

    gray = (0.75, 0.75, 0.75)
    graydark = (0.5, 0.5, 0.5)
    ns = 120

    labels = dict(zip(nodes, [g.node[node]['word'].split(',')[0] for node in nodes]))
    leaf_nodes = np.array(graph['nodes'])[graph['heights'] == 0]
    for node in leaf_nodes:
        labels[node] = ''

    cmap = plt.get_cmap('Oranges')
    fig = plt.figure(figsize=(20, 5.5))
    num_budget_points = 4
    for i, b in enumerate(np.linspace(0, len(budgets)-1, num_budget_points).astype('int')):
        for h in np.unique(graph['heights']):
            ax = fig.add_subplot(1, num_budget_points, i+1)
            h_nodes = np.array(graph['nodes'])[graph['heights'] == h].tolist()
            c = np.minimum(1, height_portions.mean(1)[h][b] / .5)
            nx.draw_networkx_nodes(g, pos, nodelist=h_nodes, node_size=ns, node_color=cmap(c), ax=ax).set_edgecolor(graydark)
            if i == 0:
                if h > 0:
                    pos_offset = pos.copy()
                    for k in pos_offset.keys():
                        pos_offset[k] = (pos_offset[k][0], pos_offset[k][1]-20)
                    nx.draw_networkx_labels(g, pos_offset, labels, font_size=12, font_color=graydark)
        nx.draw_networkx_edges(g, pos, arrows=False, edge_color=[gray] * len(g.edges()), ax=ax)
        ax.set_title('Budget: {}'.format(budgets[b]))
        plt.axis('off')
        plt.axis('equal')
    fig.tight_layout()

    fig = plt.figure()
    cmap = plt.cm.winter
    for h in range(height_portions.shape[0]):
        plt.plot(height_portions[h, 2, :], label='height {}'.format(h), color=cmap(1. * h / height_portions.shape[0]))
    plt.legend()


def iterative_missing_data(
        imagenet, accuracy_guarantees, filling_method, lambdas_method):
    """
    Parameters
    ----------
    imagenet : tc.data_source.Imagenet object

    accuracy_guarantees: list of float
        Accuracy guarantees to compute hedging thresholds for.

    filling_method : string in ['0', 'mean', 'smoothed']

    lambdas_method: string in ['original', 'empirical']
        - original: train lambdas on full, original validation data
        - empirical: corrupt the validation data in ways that test data will be
            corrupted, and learn lambdas on that
    """
    assert(filling_method in ['0', 'mean', 'smoothed'])
    assert(lambdas_method in ['original', 'empirical'])

    def apply_budget_to_probs(probs, budget, filling_method=filling_method):
        probs = probs.copy()
        mask = np.random.rand(*probs.shape) > budget
        if filling_method == '0':
            probs[mask] = 0
        elif filling_method == 'mean':
            probs[mask] = prior[mask]
        elif filling_method == 'smoothed':
            raise("Not implemented!")
            # TODO
        return probs

    num_budgets = 5  # TODO increase this for fine-grainedness
    budgets = np.linspace(0, 1, num_budgets)

    if filling_method == 'mean':
        prior = np.tile(np.mean(imagenet.X, 0), (imagenet.X_test.shape[0], 1))

    confidence = .95
    num_bs_iters = 30
    if lambdas_method == 'original':
        lambdas = darts_bisection(imagenet.X, accuracy_guarantees,
                                  imagenet.y, imagenet.graph, num_bs_iters, confidence)
    elif lambdas_method == 'empirical':
        # TODO: introduce sampling here?
        noisy_val_leaf_probs = np.vstack([apply_budget_to_probs(imagenet.X, budget) for budget in budgets])
        lambdas = darts_bisection(noisy_val_leaf_probs, accuracy_guarantees,
                                  np.tile(imagenet.y, (len(budgets), 1)), imagenet.graph, num_bs_iters, confidence)

    r, a, hp, ha = zip(*[
        darts_eval(apply_budget_to_probs(imagenet.X_test, budget), imagenet.y_test, lambdas, imagenet.graph)
        for budget in budgets])
    rewards = np.array(r)
    accuracies = np.array(a)
    height_portions = np.dstack(hp)
    height_accs = np.dstack(ha)

    # add the rewards = 0, accuracies = 1 point to all curves
    rewards = np.hstack((rewards, np.zeros((rewards.shape[0], 1))))
    accuracies = np.hstack((accuracies, np.ones((accuracies.shape[0], 1))))
    return budgets, rewards, accuracies, height_portions, height_accs
