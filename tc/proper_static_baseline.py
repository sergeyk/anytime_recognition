"""
Baseline to select classifiers based on information gain, solved as a DP
problem without any MDP machinery.
The goal is to show that the MDP comes out with the same solution, and
to have a way to initialize policies.
"""
import optparse
import sys
import os
import json
import sklearn
from sklearn.cross_validation import train_test_split
import bottleneck as bn
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.distributions
import cPickle as pickle
from scipy.interpolate import interp1d
import tc


def get_states_from_mask_distribution(ds, md, instances, labels, N, mi=None):
    state = tc.TimelyState(ds.action_dims)
    mask = md.sample(N)
    ind = np.random.choice(instances.shape[0], N)
    instances = instances[ind]
    costs = np.dot(mask, ds.action_costs)
    states = state.get_states_from_mask(instances, mask, costs)

    # Impute unobserved values.
    if mi is not None:
        states = mi.impute(states)

    labels = labels[ind]
    return states, labels


def get_states(ds, instances, action_inds, mi=None):
    state = tc.TimelyState(ds.action_dims)
    cost = ds.action_costs[action_inds].sum() / ds.max_budget
    assert(cost <= 1)
    N = instances.shape[0]
    states = np.zeros((N, state.S))
    for i in xrange(N):
        states[i] = state.get_state(instances[i], action_inds, cost)

    # Impute unobserved values.
    if mi is not None:
        states = mi.impute(states)

    return states


def get_classifier(ds, states, labels, num_clf, num_workers):
    clf = tc.StateClassifier(ds.action_dims, len(ds.labels), num_clf)
    score, entropy = clf.fit(states, labels, num_workers, verbose=False)
    return clf, score, entropy


def eval_classifier(clf, states, labels):
    proba = clf.predict_proba(states)
    score = sklearn.metrics.accuracy_score(proba.argmax(1), labels)
    entropy = (scipy.stats.distributions.entropy(proba.T) / np.log(proba.shape[1])).mean()
    if np.isnan(entropy):
        entropy = 1
    return score, entropy


def plot_dp(m, actions, title=None, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(m, vmin=0, vmax=1, cmap=plt.cm.gray)
    ax.set_xlabel('Step')
    ax.set_ylabel('Action')
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions)
    if title is not None:
        ax.set_title(title)
    for i in xrange(0, m.shape[0]):
        for j in xrange(0, m.shape[1]):
            val = m[i, j]
            if np.isnan(val):
                continue
            if val < 0.5:
                ax.text(j - 0.2, i + 0.1, '%.2f' % val, color='w')
            else:
                ax.text(j - 0.2, i + 0.1, '%.2f' % val, color='k')
    if filename is not None:
        plt.savefig(filename, dpi=300)
    return fig


class StaticClassifier(object):
    """
    Finds policy with exact classifiers.

    Parameters
    ----------
    clf_method: string in ['logreg', 'imagenet']
    """
    def __init__(self, ds, clf_method='logreg', log_dirname='data/timely_results'):
        self.name = 'static_classifier'
        assert(clf_method in ['logreg', 'imagenet'])
        self.clf_method = clf_method
        ds.validate()
        self.ds = ds
        self.logging_dirname = log_dirname + '/' + self.ds.name + '/' + self.name
        tc.util.mkdir_p(self.logging_dirname)
        self.filename = self.logging_dirname + '/sc.pickle'
        self.has_been_fit = False
        self.num_clf = 1

    def __repr__(self):
        return 'StaticClassifier'

    def save(self):
        """
        Save self to canonical location.
        """
        with open(self.filename, 'w') as f:
            pickle.dump(self, f, protocol=2)
        return self.filename

    def evaluate(self, save_plot=True, force=False):
        """
        Assemble something that I can store and use alongside what
        evaluation.evaluate_performance method uses, which is a list of
        interpolated points and their average loss.
        """
        if not self.has_been_fit:
            raise Exception("Have to fit first.")

        data_filename = self.logging_dirname + '/evaluation_final.npz'
        if not force and os.path.exists(data_filename):
            print('Evaluation already exists, not running')
            return

        costs, errors = self.predict()
        num_interp_points = ds.max_budget * 2
        interp_points = np.linspace(0, self.ds.max_budget, num_interp_points)
        f = interp1d(costs, errors, bounds_error=False, fill_value=errors[-1])
        means = f(interp_points)
        auc = round(sklearn.metrics.auc(interp_points, means) / ds.max_budget, 3)
        final = round(means[-1], 3)

        # Save stuff to make it loadable by aggregate_results.
        np.savez(data_filename, interp_points=interp_points, means=means)

        report = {}
        report['info'] = {
            'data_source': self.ds.__config__(),
            'policy_method': 'dp',
            'clf_method': self.clf_method,
            'impute_method': self.impute_method,
            'num_clf': self.num_clf
        }
        report['eval'] = {
            'perf': {'loss_auc': auc, 'loss_final': final}
        }
        report_filename = self.logging_dirname + '/report.json'
        with open(report_filename, 'w') as f:
            json.dump(report, f)

        if save_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(interp_points, means, '-')
            ax.plot(costs, errors, 's')
            ax.set_xlabel('Cost')
            ax.set_xlim([0, ds.max_budget])
            ax.set_ylabel('Error')
            ax.set_ylim([0, 1])
            filename = self.logging_dirname + '/eval.png'
            plt.savefig(filename)

    def predict(self):
        clfs = self.clfs
        action_inds = self.action_inds
        instances = ds.X_test
        labels = ds.y_test

        errors = []
        costs = []
        for i in xrange(len(action_inds) + 1):
            states = get_states(ds, instances, action_inds[:i])
            errors.append(1 - clfs[i].score(states, labels))
            costs.append(ds.action_costs[action_inds[:i]].sum())
        return costs, errors

    def fit(self, num_workers=7, debug_plots=True, force=False):
        """
        Train A+1 classifiers (the first one is for no features observed.)
        Find order of A features to run in.
        """
        if not force and os.path.exists(self.filename):
            with open(self.filename) as f:
                sc = pickle.load(f)
            self.action_inds = sc.action_inds
            self.clfs = sc.clfs
            self.has_been_fit = True
            return

        ds = self.ds
        instances = ds.X
        labels = ds.y
        A = len(ds.actions)

        # Train classifier on initially empty states.
        action_inds = []
        states = get_states(ds, instances, action_inds)
        clf, score, entropy = get_classifier(
            ds, states, labels, 1, num_workers)

        # We will collect values for visualization.
        scores = np.empty((A, A))
        scores.fill(np.nan)
        entropies = np.empty((A, A))
        entropies.fill(np.nan)
        infogains = np.empty((A, A))
        infogains.fill(np.nan)

        # While there are feasible actions, consider them.
        costs = ds.action_costs.copy()
        remaining_mask = np.ones(len(ds.actions), dtype=bool)
        selected_clfs = [clf]
        for iteration in xrange(A):
            print('-'*80)
            print('Iteration {}'.format(iteration))

            feas_inds = np.flatnonzero(remaining_mask & (costs <= ds.max_budget))
            if len(feas_inds) == 0:
                break

            new_clfs = np.empty(A, dtype=object)
            for action_ind in feas_inds:
                print(ds.actions[action_ind]),
                new_action_inds = action_inds + [action_ind]
                states = get_states(ds, instances, new_action_inds)

                new_clf, new_score, new_entropy = get_classifier(
                    ds, states, labels, 1, num_workers)
                new_clfs[action_ind] = new_clf

                infogains[action_ind, iteration] = entropy - new_entropy
                scores[action_ind, iteration] = new_score
                entropies[action_ind, iteration] = new_entropy

            rewards = infogains[:, iteration] / ds.action_costs
            ind = bn.nanargmax(rewards)
            selected_clfs.append(new_clfs[ind])
            action_inds.append(ind)
            print('Selected {} with infogain {:.2f} and cost {:.2f}'.format(ds.actions[ind], infogains[ind, iteration], ds.action_costs[ind]))

            remaining_mask[ind] = False
            costs += ds.action_costs[ind]
            entropy = entropies[ind, iteration]

        actions = np.take(ds.actions, action_inds)
        print('Selected actions in order: {}'.format(actions))

        self.action_inds = action_inds
        self.clfs = selected_clfs
        assert(len(self.clfs) == len(self.action_inds) + 1)
        self.has_been_fit = True
        if debug_plots:
            self.plot_stuff(scores, entropies, infogains, rewards)
        self.save()

    def plot_stuff(self, scores, entropies, infogains, rewards):
        ds = self.ds
        A = len(ds.actions)
        rewards = infogains / np.tile(ds.action_costs, (A, 1)).T
        plot_dp(scores, ds.actions, title='Accuracy',
                filename=self.logging_dirname + '/debug_accuracy.png')
        plot_dp(entropies, ds.actions, title='Entropy',
                filename=self.logging_dirname + '/debug_entropy.png')
        plot_dp(infogains, ds.actions, title='Infogain',
                filename=self.logging_dirname + '/debug_infogain.png')
        plot_dp(rewards, ds.actions, title='Reward',
                filename=self.logging_dirname + '/debug_reward.png')


class StaticClassifierClustered(StaticClassifier):
    def __init__(self, ds, clf_method='logreg', impute_method='mean',
                 num_clf=1, log_dirname='data/timely_results'):
        ds.validate()
        self.ds = ds

        self.num_clf = num_clf

        assert(clf_method in ['logreg', 'imagenet'])
        self.clf_method = clf_method

        assert(impute_method in ['mean', 'gaussian'])
        self.impute_method = impute_method

        self.name = 'static_classifier_clustered_{}_{}_{}'.format(
            self.num_clf, self.clf_method, self.impute_method)

        self.logging_dirname = log_dirname + '/' + self.ds.name + '/' + self.name
        tc.util.mkdir_p(self.logging_dirname)

        self.filename = self.logging_dirname + '/sc.pickle'
        self.has_been_fit = False

    def __repr__(self):
        return 'StaticClassifierClustered: num_clf={} clf_method={} impute_method={}'.format(
            self.num_clf, self.clf_method, self.impute_method)

    def predict(self):
        action_inds = self.action_inds
        instances = ds.X_test
        labels = ds.y_test

        errors = []
        costs = []
        for i in xrange(len(action_inds) + 1):
            states = get_states(ds, instances, action_inds[:i], self.mi)
            errors.append(1 - self.clf.score(states, labels))
            costs.append(ds.action_costs[action_inds[:i]].sum())
        return costs, errors

    def fit(self, num_workers=1, debug_plots=True, force=False):
        if not force and os.path.exists(self.filename):
            with open(self.filename) as f:
                sc = pickle.load(f)
            self.__dict__.update(sc.__dict__)
            return

        instances = ds.X
        labels = ds.y

        instances_train, instances_val, labels_train, labels_val = \
            train_test_split(instances, labels, test_size=1/3.)
        A = len(ds.actions)
        N_train = instances_train.shape[0]

        # Initialize imputation mechanism
        if self.impute_method == 'mean':
            mi = tc.MeanImputer(ds.action_dims).fit(instances_train)
        else:
            mi = tc.GaussianImputer(ds.action_dims).fit(instances_train)

        # Train classifier on initially empty states.
        action_inds = []
        states_train = get_states(ds, instances_train, action_inds, mi)
        states_val = get_states(ds, instances_val, action_inds, mi)

        if self.clf_method == 'logreg':
            clf, score_train, entropy_train = get_classifier(
                ds, states_train, labels_train, self.num_clf, num_workers)
        else:
            clf = tc.StateClassifierImagenet(ds)
        score_val, entropy_val = eval_classifier(clf, states_val, labels_val)

        # We collect values for visualization.
        scores = np.empty((A, A))
        scores.fill(np.nan)
        entropies = np.empty((A, A))
        entropies.fill(np.nan)
        infogains = np.empty((A, A))
        infogains.fill(np.nan)

        # While there are feasible actions, consider them.
        costs = ds.action_costs.copy()
        remaining_mask = np.ones(len(ds.actions), dtype=bool)
        policy_masks = [remaining_mask.copy()]
        for iteration in xrange(A):
            print('-'*80)
            print('Iteration {}'.format(iteration))

            feas_inds = np.flatnonzero(remaining_mask & (costs <= ds.max_budget))
            if len(feas_inds) == 0:
                break

            # Train classifier with new mask distribution.
            if self.clf_method != 'imagenet':
                new_masks = []
                for action_ind in feas_inds:
                    mask = remaining_mask.copy()
                    mask[action_ind] = False
                    new_masks.append(mask)
                md = tc.MaskDistribution()
                md.update(np.array(policy_masks + new_masks))
                N = N_train * ((iteration + 1) + len(feas_inds))
                states_, labels_ = get_states_from_mask_distribution(
                    ds, md, instances_train, labels_train, N, mi)
                clf, score_train, entropy_train = get_classifier(
                    ds, states_, labels_, self.num_clf, num_workers)

            # Evaluate the infogain of individual features.
            for action_ind in feas_inds:
                print(ds.actions[action_ind]),
                states_val = get_states(
                    ds, instances_val, action_inds + [action_ind], mi)
                new_score_val, new_entropy_val = eval_classifier(
                    clf, states_val, labels_val)

                infogains[action_ind, iteration] = entropy_val - new_entropy_val
                scores[action_ind, iteration] = new_score_val
                entropies[action_ind, iteration] = new_entropy_val

            rewards = infogains[:, iteration] / ds.action_costs
            ind = bn.nanargmax(rewards)
            action_inds.append(ind)

            entropy_val = entropies[ind, iteration]
            print('Selected {} with infogain {:.2f} and cost {:.2f}'.format(ds.actions[ind], infogains[ind, iteration], ds.action_costs[ind]))

            remaining_mask[ind] = False
            costs += ds.action_costs[ind]
            policy_masks += [remaining_mask.copy()]

        # Fit imputer with all data
        self.mi = mi.fit(instances)

        # Train final classifier, with the final masks and on full data
        if self.clf_method != 'imagenet':
            md = tc.MaskDistribution()
            md.update(np.array(policy_masks))
            N = N_train * len(policy_masks)
            states_, labels_ = get_states_from_mask_distribution(
                ds, md, instances, labels, N, self.mi)
            clf, score, entropy = get_classifier(
                ds, states_, labels_, self.num_clf, num_workers)

        actions = np.take(ds.actions, action_inds)
        print('Selected actions in order: {}'.format(actions))

        self.clf = clf
        self.action_inds = action_inds
        self.has_been_fit = True
        if debug_plots:
            self.plot_stuff(scores, entropies, infogains, rewards)
        self.save()


if __name__ == '__main__':
    #ds = tc.data_sources.SyntheticOrthants('data/data_sources', D=2, N=12000, N_test=3000)
    #ds = tc.data_sources.Scene15('data/data_sources', max_budget=60)

    # sc = StaticClassifier(ds)
    # sc.fit(num_workers=4)
    # sc.evaluate(save_plot=True)

    # sc = StaticClassifierClustered(ds, -1)
    # sc.fit(num_workers=1)
    # sc.evaluate(save_plot=True)

    # sc = StaticClassifierClustered(ds, 3)
    # sc.fit(num_workers=1)
    # sc.evaluate(save_plot=True)

    # ds = tc.data_sources.ILSVRC65('data/data_sources', max_budget=13)
    # ds.save()
    # sc = StaticClassifierClustered(ds, 'imagenet', impute_method='gaussian', num_clf=1)
    # sc.fit(num_workers=1)
    # sc.evaluate(save_plot=True)

    # sys.exit(0)

    usage = "usage: %prog [options] <data_source_pickle_filename>"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('--force', action="store_true", default=False)
    parser.add_option('--num_workers', type='int', default=1)
    parser.add_option('--method', default='clustered')
    parser.add_option('--num_clf', default=1)
    parser.add_option('--clf_method', default='logreg')
    parser.add_option('--impute_method', default='mean')
    opts, args = parser.parse_args()

    # Load the DataSource
    if len(args) != 1:
        parser.error("incorrect number of arguments")
    data_source_pickle_filename = args[0]
    with open(data_source_pickle_filename) as f:
        ds = pickle.load(f)

    num_workers = opts.num_workers
    force = opts.force
    if opts.method == 'exact':
        sc = StaticClassifier(ds, opts.clf_method)
    elif opts.method.split('_')[0] == 'clustered':
        sc = StaticClassifierClustered(
            ds, clf_method=opts.clf_method, num_clf=opts.num_clf,
            impute_method=opts.impute_method)
    else:
        raise Exception('do not understand method')

    print sc
    sc.fit(num_workers=1)
    sc.evaluate(save_plot=True)
