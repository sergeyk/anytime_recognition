import numpy as np
import pandas
import sklearn
import scipy.stats.distributions
import joblib
import json
import time
import tc


def fit_grid_point(X, y, X_val, y_val, base_clf, clf_params):
    clf = sklearn.clone(base_clf)
    clf.set_params(**clf_params)
    clf.fit(X, y)
    proba = clf.predict_proba(X_val)
    score = sklearn.metrics.accuracy_score(proba.argmax(1), y_val)
    entropy = (scipy.stats.distributions.entropy(proba.T) / np.log(proba.shape[1])).mean()
    return json.dumps(clf_params), score, entropy


class StateClassifierImagenet(object):
    def __init__(self, ds):
        # Need datasource because need feature name and bound information
        ds.validate()
        self.ds = ds
        self.state = tc.TimelyState(ds.action_dims)

    def plot_weights(self, filename=None):
        return None

    def fit(self, states, labels, num_workers=1, verbose=False):
        pass

    def score(self, states, labels):
        labels_pred = self.predict(states)
        return sklearn.metrics.accuracy_score(labels_pred, labels)

    def predict(self, states):
        return self.predict_proba(states).argmax(1)

    def predict_proba(self, states):
        # TODO: when we use multiple feature channels, revise this to average
        # over feature channels
        proba = self.state.slice_array(states, 'observations')
        return proba


class StateClassifier(object):
    def __init__(self, action_dims, num_labels, num_clf=1, max_masks=None):
        self.num_labels = num_labels
        self.state = tc.TimelyState(action_dims)
        self.num_clf = num_clf
        self.max_masks = max_masks
        self.has_been_fit = False

    def plot_weights(self, filename=None):
        if self.has_been_fit:
            if self.num_clf == 1:
                fig = tc.util.plot_weights(
                    self.clf.coef_, xlabel='weights on features',
                    ylabel='Classes', filename=filename)
                return [filename]
            else:
                # Need to plot multiple times.
                filenames = []
                for i, clf in enumerate(self.clfs):
                    new_filename = '{}_{}{}'.format(filename[:-4], i, filename[-4:])
                    filenames.append(new_filename)
                    fig = tc.util.plot_weights(
                        clf.coef_, xlabel='Weights on features',
                        ylabel='Classes', filename=new_filename)
                return filenames
        else:
            return None

    def score(self, states, labels):
        labels_pred = self.predict(states)
        return sklearn.metrics.accuracy_score(labels_pred, labels)

    def predict(self, states):
        return self.predict_proba(states).argmax(1)

    def _fit(self, X, y, num_workers=1, verbose=False):
        t = time.time()

        param_grid = [{
            'C': [.001, .1, 10, 100],
        }]
        base_clf = sklearn.linear_model.LogisticRegression(
            fit_intercept=False, class_weight='auto',
            dual=False, penalty='l2')

        kfold = sklearn.cross_validation.StratifiedKFold(y, 3)
        param_iterator = sklearn.grid_search.ParameterGrid(param_grid)
        out = joblib.Parallel(n_jobs=num_workers, pre_dispatch=num_workers * 2)(
            joblib.delayed(fit_grid_point)(
                X[train], y[train], X[test], y[test], base_clf, clf_params)
            for clf_params in param_iterator for train, test in kfold
        )

        df = pandas.DataFrame(out, columns=['setting', 'score', 'entropy'])
        dfg = df.groupby('setting').mean()
        if verbose:
            print(dfg)

        dfg = dfg.sort(['score', 'entropy'], ascending=[0, 1])
        best_params = json.loads(dfg.index[0])
        best_score, best_entropy = dfg.ix[0].values
        print('Best at {}: {:.2f} | {:.2f} and took {:.2f} s'.format(best_params, best_score, best_entropy, time.time() - t))

        clf = sklearn.clone(base_clf)
        clf.set_params(**best_params)
        clf.fit(X, y)
        return clf, best_score, best_entropy

    def fit(self, states, labels, num_workers=1, verbose=False):
        """
        Fit one classifier to the data.
        """
        if self.num_clf != 1:
            return self.fit_clustered(states, labels, num_workers, verbose)

        X = np.hstack((
            self.state.slice_array(states, 'observations'),
            self.state.slice_array(states, 'bias')
        ))
        y = labels
        self.clf, best_score, best_entropy = self._fit(
            X, y, num_workers, verbose)
        self.has_been_fit = True
        return best_score, best_entropy

    def fit_clustered(self, states, labels, num_workers=1, verbose=False):
        """
        Fit a number of classifiers to the data, clustered by state masks.
        """
        X = np.hstack((
            self.state.slice_array(states, 'observations'),
            self.state.slice_array(states, 'bias')
        ))
        y = labels

        mask = self.state.get_mask(states).astype(bool)

        self.md = tc.MaskDistribution(self.max_masks)
        self.md.update(mask)
        cluster_ind = self.md.predict_cluster(mask, self.num_clf)
        unique_inds = np.unique(cluster_ind)
        self.clfs = []
        for ind in sorted(unique_inds):
            umask = self.md.umasks[ind]
            print(umask),
            try:
                clf, best_score, best_entropy = self._fit(
                    X[cluster_ind == ind], y[cluster_ind == ind],
                    num_workers, verbose)
            except Exception as e:
                print(e)
                clf, best_score, best_entropy = self._fit(
                    X, y, num_workers, verbose)
            self.clfs.append(clf)

        self.has_been_fit = True
        return best_score, best_entropy

    def predict_proba(self, states):
        if self.num_clf != 1:
            return self.predict_proba_clustered(states)

        if not self.has_been_fit:
            raise Exception('StateClassifier not been fit yet.')
        X = np.hstack((
            self.state.slice_array(states, 'observations'),
            self.state.slice_array(states, 'bias')
        ))
        return self.clf.predict_proba(X)

    def predict_proba_clustered(self, states):
        if not self.has_been_fit:
            raise Exception('StateClassifier not been fit yet.')
        X = np.hstack((
            self.state.slice_array(states, 'observations'),
            self.state.slice_array(states, 'bias')
        ))

        mask = self.state.get_mask(states).astype(bool)
        cluster_ind = self.md.predict_cluster(mask, self.num_clf)

        proba = np.empty((states.shape[0], self.num_labels))
        unique_inds = np.unique(cluster_ind)
        for ind in sorted(unique_inds):
            clf = self.clfs[ind]
            proba[cluster_ind == ind] = clf.predict_proba(X[cluster_ind == ind])
        return proba
