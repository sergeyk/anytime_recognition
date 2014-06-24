import numpy as np
import cPickle as pickle
import optparse
import os
import json
from sklearn.metrics import accuracy_score
import tc


class SingleClassifier(object):
    """
    Baseline to train single-feature classifiers and output
    - classifier trained on all features
    - classifier trained on the fastest feature
    - classifier trained on the most accurate feature
    - classifier trained on the highest value feature

    Writes out results to log_dirname/{ds.budgetless_name}/{clf_method}/all.json

    Parameters
    ----------
    # TODO
    """
    def __init__(self, data_source, log_dirname, clf_method='logreg'):
        self.ds = data_source

        if clf_method == 'logreg':
            self.clf_str = 'tc.classifier.LogisticClassifier(ds, 1)'
        elif clf_method == 'gnb':
            self.clf_str = 'tc.classifier.GaussianNBClassifier(ds, 1)'
        else:
            raise Exception("clf_method {} is not implemented!".format(clf_method))
        self.clf_method = clf_method

        #log_dirname = os.path.relpath(log_dirname, tc.repo_dir)
        logging_dirname = log_dirname + '/' + self.ds.budgetless_name
        tc.util.mkdir_p(logging_dirname)
        self.filename = logging_dirname + '/{}.json'.format(self.clf_method)

    def run(self, num_workers=1, force=False):
        ds = self.ds
        results = {}

        if os.path.exists(self.filename) and not force:
            print("Already output single-classifier results, not running.")
            print(self.filename)
            return

        # train classifier on all features
        mask = np.zeros_like(ds.X, dtype=bool)
        clf = eval(self.clf_str)
        val_score = clf._fit(ds.X, mask, ds.y, num_workers, fit_intercept=True)
        score = accuracy_score(clf._predict_proba(
            ds.X_test).argmax(1), ds.y_test)
        print('All features: {:.3f}/{:.3f}'.format(val_score, score))
        results['all_error'] = 1 - score
        results['all_cost'] = ds.action_costs.sum()

        # train classifier per feature
        scores = []
        clfs = []
        for action in ds.actions:
            X = tc.util.slice_array(ds.X, ds.feature_bounds, action)
            mask = np.zeros_like(X, dtype=bool)
            clf = eval(self.clf_str)
            scores.append(clf._fit(X, mask, ds.y, num_workers, fit_intercept=True))
            clfs.append(clf)
        scores = np.array(scores)

        # fastest feature (with best accuracy)
        least_cost_ind = np.flatnonzero(
            ds.action_costs == ds.action_costs.min())
        least_cost_best_ind = least_cost_ind[scores[least_cost_ind].argmax()]
        action = ds.actions[least_cost_best_ind]
        X_test = tc.util.slice_array(ds.X_test, ds.feature_bounds, action)
        score = accuracy_score(clfs[least_cost_best_ind]._predict_proba(
            X_test).argmax(1), ds.y_test)
        print('Least cost best feature: {} {:.3f}/{:.3f}, cost {:.3f}'.format(
            ds.actions[least_cost_best_ind], scores[least_cost_best_ind],
            score, ds.action_costs[least_cost_best_ind]))
        results['least_cost_feature'] = ds.actions[least_cost_best_ind]
        results['least_cost_feature_cost'] = ds.action_costs[least_cost_best_ind]
        results['least_cost_feature_error'] = 1 - score

        # feature with best accuracy
        best_ind = scores.argmax()
        action = ds.actions[best_ind]
        X_test = tc.util.slice_array(ds.X_test, ds.feature_bounds, action)
        score = accuracy_score(clfs[best_ind]._predict_proba(
            X_test).argmax(1), ds.y_test)
        print('Best feature: {} {:.3f}/{:.3f}, cost {:.3f}'.format(
            ds.actions[best_ind], score, scores[best_ind],
            ds.action_costs[best_ind]))
        results['best_feature'] = ds.actions[best_ind]
        results['best_feature_cost'] = ds.action_costs[best_ind]
        results['best_feature_error'] = 1 - scores[best_ind]

        with open(self.filename, 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')

    usage = "usage: %prog [options] <data_source_pickle_filename>"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('--clf_method', default='logreg')
    parser.add_option('--log_dirname', default='data/timely_results')
    parser.add_option('--force', action="store_true", default=False)
    parser.add_option('--num_workers', type='int', default=1)
    opts, args = parser.parse_args()

    # Load the DataSource
    if len(args) != 1:
        parser.error("incorrect number of arguments")
    data_source_pickle_filename = args[0]
    with open(data_source_pickle_filename) as f:
        ds = pickle.load(f)

    num_workers = opts.num_workers
    force = opts.force
    log_dirname = opts.log_dirname
    clf_method = opts.clf_method
    opts = dict((k, v) for (k, v) in opts.__dict__.iteritems() if v is not None)

    clf = SingleClassifier(ds, log_dirname, clf_method)
    clf.run(num_workers, force)
