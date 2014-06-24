from glob import glob
import sys
import jinja2
import os
import pandas
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict


def get_experiment_results(dirname):
    """
    Assemble a DataFrame from the results of an experiment, which are
    present in a single directory.

    Returns
    -------
    max_budget: float
    experiment_df: pandas.DataFrame
    """
    reports = glob('{}/*/report.json'.format(dirname))
    info_df = pandas.DataFrame()
    perf_df = pandas.DataFrame()
    max_budgets = []
    for i, report in enumerate(reports):
        with open(report) as f:
            data = json.load(f)
        max_budgets.append(data['info']['data_source']['max_budget'])
        if 'perf' in data['eval']:
            df = pandas.DataFrame(data['info'], columns=[
                'max_iter', 'max_batches', 'batch_size',
                'policy_feat', 'policy_method',
                'rewards_mode', 'rewards_loss', 'gamma',
                'clf_method', 'num_clf', 'random_start', 'impute_method',
                'normalize_reward_locally', 'add_fully_observed'], index=[i])
            df['dirname'] = os.path.dirname(report)
            df['link'] = '<a href="{}">report</a>'.format(os.path.basename(os.path.dirname(report)) + '.html')
            info_df = info_df.append(df)

            df = pandas.DataFrame(data['eval']['perf'], index=[i])
            perf_df = perf_df.append(df)
    print('Found {} tried experimental conditions, with {} successfully evaluated.'.format(len(reports), perf_df.shape[0]))
    assert(all(x == max_budgets[0] for x in max_budgets))
    max_budget = max_budgets[0]
    experiment_df = info_df.join(perf_df)
    return max_budget, experiment_df


def filter_best_results(experiment_df, select_lowest='loss_auc'):
    """
    Return DataFrame containing the best results for each one of the
    listed settings (defined in the code below) in the experiment_df.

    Returns:
    best_df: pandas.DataFrame
    """
    settings = OrderedDict([
        ('Optimal', {'policy_method': 'manual_orthants'}),
        ('Random', {'policy_method': 'random'}),
        ('Static, greedy', {'policy_feat': 'static', 'gamma': 0}),
        ('Static, DP greedy', {'policy_method': 'dp'}),
        ('Static, non-myopic', {'policy_feat': 'static', 'gamma': 1}),
        ('Dynamic, greedy', {'policy_feat': 'dynamic', 'gamma': 0}),
        ('Dynamic, non-myopic', {'policy_feat': 'dynamic', 'gamma': 1}),
    ])
    df = experiment_df  # just shortening the name
    best_inds = []
    indices = []
    for setting, conditions in settings.items():
        try:
            mask = reduce(
                lambda a, b: a & b,
                (df[k] == v for k, v in conditions.iteritems())
            )
            if mask.sum() > 0:
                best_inds.append(df[mask][select_lowest].argmin())
                indices.append(setting)
        except Exception as e:
            print(e)
    best_df = df.ix[best_inds]
    best_df.index = indices
    return best_df


def plot_single_results(single_df, ax):
    # TODO: hackily, just use logreg here instead of determining the best
    # classifier.
    if 'logreg' in single_df.index:
        cost = single_df.ix['logreg']['all_cost']
        ax.plot([-1000, cost],
                [single_df.ix['logreg']['all_error']] * 2, 's--',
                markersize=7, linewidth=3,
                label='All features (cost {:.1f})'.format(cost))

        # cost = single_df.ix['logreg']['best_feature_cost']
        # ax.plot([-1000, cost],
        #         [single_df.ix['logreg']['best_feature_error']] * 2, 's--', markersize=7, label='Best feature (cost {:.1f})'.format(cost))

        # cost = single_df.ix['logreg']['least_cost_feature_cost']
        # ax.plot([-1000, cost],
        #         [single_df.ix['logreg']['least_cost_feature_error']] * 2, 's--', markersize=7, label='Least cost feature (cost {:.1f})'.format(cost))


def plot_results(
        df, max_budget, single_df=None,
        fontsize=20, figsize=(20, 6), filename=None, nonincreasing=False,
        ylim=[0, 1]):
    """
    Plot curves of Error vs. Cost and and bar chart of their AUCs on a
    1x2 subplot figure, optionally writing it to filename.
    """
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    plt.rc('text', usetex=True)

    cycle = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    plt.rc('axes', labelsize=fontsize, color_cycle=cycle)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('legend', fontsize=fontsize)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(121)
    for index, row in df.iterrows():
        data = np.load(row['dirname'] + '/evaluation_final.npz')
        interp_points = data['interp_points']
        means = data['means']

        if nonincreasing:
            for i in range(1, means.shape[0]):
                means[i] = min(means[i - 1], means[i])

        ax.plot(interp_points, means, 's-', label=index,
                markeredgecolor='none', markersize=5, linewidth=3)

    if single_df is not None:
        plot_single_results(single_df, ax)

    ax.vlines(max_budget, 0, 1)
    ax.set_xlabel('Cost')
    ax.set_xlim([0, max_budget + 1])
    ax.set_ylabel('Error')
    ax.set_ylim(ylim)
    plt.legend(ncol=3, loc='upper center',
               fancybox=True, shadow=True, bbox_to_anchor=(0.5, 1.15))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax2 = fig.add_subplot(122)
    # insert line breaks in labels: 'Dynamic, greedy' => 'Dynamic,\ngreedy'
    # to make sure they fit in horizontal orientation.
    df.index = ['\n'.join(x.split(' ')) for x in df.index]
    df['Area under Error vs. Cost curve'] = df['loss_auc']
    df['Final Error'] = df['loss_final']
    df[['Area under Error vs. Cost curve', 'Final Error']].plot(
        ax=ax2, kind='bar', color=cycle, rot=0)

    # print the numbers on the bars
    # patches are plotted first all AUC then all Final, not intertwined
    N = len(df['loss_auc'])
    for i, auc, final in zip(range(N), df['loss_auc'], df['loss_final']):
        rect = ax2.patches[i]
        plt.text(rect.get_x() + rect.get_width() / 2.,
                 max(.015, rect.get_height() - .05), '{:.3f}'.format(auc),
                 ha='center', va='bottom', color='white', fontsize=.7 * fontsize)
        rect = ax2.patches[N + i]
        plt.text(rect.get_x() + rect.get_width() / 2.,
                 max(.015, rect.get_height() - .05), '{:.3f}'.format(final),
                 ha='center', va='bottom', color='black', fontsize=.7 * fontsize)

    for item in ax2.get_xticklabels():
        item.set_fontsize(fontsize)

    ax2.set_ylim([0, .85])

    for spine in ax2.spines.itervalues():
        spine.set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')

    fig.subplots_adjust(bottom=0.15)
    if filename is not None:
        plt.savefig(filename)
    return fig


def plot_scenes_vs_budgets(name, figsize=(9, 6), fontsize=16):
    """
    Display all Scenes15 results on one plot.
    """
    if name == 'scene15':
        dirnames = glob('data/timely_results/scene15_*')
        single_df = get_single_clf_results('data/timely_results/scene15')
        filename = 'data/timely_results/_scenes15'
    elif name == 'ilsvrc65':
        dirnames = glob('data/timely_results/ilsvrc65_*')
        single_df = get_single_clf_results('data/timely_results/ilsvrc65')
        filename = 'data/timely_results/_ilsvrc65'
    else:
        raise Exception('Unknown name.')

    dfs = {}
    for dirname in dirnames:
        mb = int(dirname.split('_')[-1])
        if (name == 'scene15' and mb in [1]):
            continue
        dfs[mb] = pandas.read_pickle(dirname + '/best.df')

    panel = pandas.Panel.from_dict(dfs)

    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    plt.rc('text', usetex=True)

    plt.rc('axes', labelsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('legend', fontsize=fontsize)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # TODO: acceptable?
    # make values non-increasing
    vals = panel.loc[:, :, 'loss_auc'].values
    for i in range(vals.shape[0]):
        for j in range(1, vals.shape[1]):
            vals[i, j] = min(vals[i, j - 1], vals[i, j])
    final_vals = panel.loc[:, :, 'loss_final'].values
    for i in range(final_vals.shape[0]):
        for j in range(1, final_vals.shape[1]):
            final_vals[i, j] = min(final_vals[i, j - 1], final_vals[i, j])

    ordered_columns = [
        'Random', 'Static, greedy', 'Static, non-myopic',
        'Dynamic, greedy', 'Dynamic, non-myopic']

    df = panel.loc[:, :, 'loss_auc'].T
    df = df[ordered_columns]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    df.plot(ax=ax, marker='s', linewidth=3, color=colors)
    ax.set_xlabel('Max Budget')
    ax.set_ylabel('Area under the Error vs. Cost curve')
    if name == 'ilsvrc65':
        ax.set_xlim((2, 60))
        ax.set_ylim((.73, 1))
    if name == 'scene15':
        ax.set_xlim((1, 30))
        ax.set_ylim((.1, .7))
    # ax.set_xlim(left=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.legend(ncol=1, fancybox=True, shadow=False)
    plt.savefig(filename + '_auc.png', dpi=300)
    plt.savefig(filename + '_auc.pdf')

    #### 2nd figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plot_single_results(single_df, ax)

    df = panel.loc[:, :, 'loss_final'].T
    df = df[ordered_columns]

    print df.to_string()

    df.plot(
        ax=ax, marker='s', linewidth=3, color=colors)
    ax.set_xlabel('Max Budget')
    ax.set_ylabel('Final Error')
    if name == 'ilsvrc65':
        ax.set_xlim((2, 60))
        ax.set_ylim((.6, 1))
    if name == 'scene15':
        ax.set_xlim((1, 30))
    plt.legend(ncol=1, fancybox=True, shadow=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.savefig(filename + '_final.png', dpi=300)
    plt.savefig(filename + '_final.pdf')


jt = jinja2.Template("""
<html>
<head>
<style>
body {
    font-family: monospace;
}
table {
    width:90%;
    margin: 0.5em auto;
    border: none;
}
td, th {
    padding: .3em 1em;
    border: none;
}
tr:nth-child(odd) {
    background: #cccccc;
}
</style>
</head>
<body>
<div>
    <h3>Plots</h3>
    <img src="_summary.png" width="90%" />
</div>
<hr />
<div>
    <h3>Best result in category</h3>
    {{ best }}
</div>
<hr />
<div>
    <h3>All results</h3>
    {{ experiment }}
</div>
</body>
</html>
""")


def get_single_clf_results(dirname):
    """
    Return a (potentially empty) DataFrame with single-clf results.
    """
    results = {}
    filenames = glob(dirname + '/*.json')
    for filename in filenames:
        with open(filename) as f:
            data = json.load(f)
        clf_method = os.path.basename(filename)[:-5]
        results[clf_method] = data
    df = pandas.DataFrame(results)
    return df.T


if __name__ == '__main__':
    """
    If there is a command line argument, it is the name of a directory
    to process. Otherwise, process all experiments.
    """
    if len(sys.argv) > 1:
        print("here")
        experiment_dirnames = [sys.argv[1]]
    else:
        experiment_dirnames = glob('data/timely_results/*')

    for dirname in experiment_dirnames:
        print(dirname)
        try:
            max_budget, experiment_df = get_experiment_results(dirname)
        except:
            print("Directory could not be processed.")
            continue
        best_df = filter_best_results(experiment_df)
        best_df.save(dirname + '/best.df')

        html_filename = dirname + '/_summary.html'
        with open(html_filename, 'w') as f:
            experiment_df_print = experiment_df.copy()
            del experiment_df_print['dirname']
            experiment_df_print.columns = [' '.join(c.split('_')) for c in experiment_df_print.columns]
            best_df_print = best_df.copy()
            del best_df_print['dirname']
            best_df_print.columns = [' '.join(c.split('_')) for c in best_df_print.columns]

            pandas.set_option('max_colwidth', 250)
            f.write(jt.render(
                experiment=experiment_df_print.to_html(escape=False),
                best=best_df_print.to_html(escape=False)))

        # the budget is after the last underscore, so just take everything
        # except that to get the budgetless name.
        budgetless_dirname = '_'.join(dirname.split('_')[:-1])
        try:
            single_df = get_single_clf_results(budgetless_dirname)
        except:
            single_df = None

        plot_filename = dirname + '/_summary.png'
        fig = plot_results(
            best_df, max_budget, single_df, 20, (22, 6), plot_filename, True)

    # plot_scenes_vs_budgets('scene15')
    # plot_scenes_vs_budgets('ilsvrc65')
