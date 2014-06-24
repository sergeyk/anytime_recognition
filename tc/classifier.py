import tc
from hurry.filesize import size as hsize
import sklearn.linear_model
import sklearn.ensemble
import sklearn.naive_bayes
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
import abc
import numpy as np


class PredictorClassifier(object):
    """
    A Predictor-based classifier has a self.clf object that implements
    fit() and predict_proba() methods.

    Parameters
    ----------
    ds: tc.DataSource
    """
    def __init__(self, ds, num_clf):
        self.ds = ds
        self.num_clf = num_clf
        self.num_classes = len(ds.labels)
        self.state = tc.TimelyState(self.ds.action_dims)
        self.has_been_fit = False

    def __repr__(self):
        clf = self.clf if hasattr(self, 'clf') else 'not fit yet'
        return '{}: {}'.format(
            self.__class__.__name__, clf)

    def plot_weights(self, filename=None):
        if hasattr(self.clf, 'coef_'):
            return tc.util.plot_weights(
                self.clf.coef_, xlabel='weights on features',
                ylabel='Classes', filename=filename)
        else:
            return None

    __metaclass__ = abc.ABCMeta

    def fit(self, states_arr, states_labels, instances, labels, val_states_arr,
            val_states_labels, add_fully_observed=False, num_workers=1):
        """
        Train on state data obtained by the policy, as well as, optionally,
        the training set from which the policy drew samples.

        Parameters
        ----------
        states_arr: (M, S) ndarray of float
        states_labels: (M,) ndarray of int
        instances: (N, D) ndarray of float
        labels: (N,) ndarray of int
        val_states_arr: (M', S) ndarray of float
        val_states_labels: (M',) ndarray of int
        add_fully_observed: bool, optional [False]
            If True, also train on fully observed data from instances.
        num_workers: int

        Returns
        -------
        best_score: float
            Best score obtained during the parameter search.
        """
        # remove the fully observed and fully unobserved states
        states_masks = self.state.get_mask(states_arr).astype(bool)
        F = states_masks.shape[1]
        ind = (states_masks == np.ones(F, dtype=bool)).all(1)
        ind |= (states_masks == np.zeros(F, dtype=bool)).all(1)

        states_arr = states_arr[~ind]
        mask = states_masks[~ind]
        feature_mask = self.state.get_feature_mask(mask)
        states_labels = states_labels[~ind]

        X = np.hstack((
            self.state.slice_array(states_arr, 'observations'),
            self.state.slice_array(states_arr, 'bias')
        ))
        y = states_labels
        print('Classifier: {} not-fully-unobserved states.'.format(X.shape[0]))

        if add_fully_observed:
            # add at most half as much fully observed data
            N = instances.shape[0]
            if N > X.shape[0] / 2:
                ind = np.random.choice(N, X.shape[0] / 2)
                fully_observed = np.ones((X.shape[0] / 2, X.shape[1]))
                fully_observed[:, :-1] = instances[ind]
                fully_observed_labels = labels[ind]
            else:
                fully_observed = np.ones((N, X.shape[1]))
                fully_observed[:, :-1] = instances
                fully_observed_labels = labels
            X = np.vstack((X, fully_observed))
            feature_mask = np.vstack((feature_mask, np.zeros((fully_observed.shape[0], self.state.D), dtype=bool)))
            y = np.hstack((y, fully_observed_labels))

        assert(X.shape[0] == feature_mask.shape[0])
        assert(X.shape[1] == feature_mask.shape[1] + 1)
        print("X size is {}".format(hsize(X.nbytes)))

        X_val = np.hstack((
            self.state.slice_array(val_states_arr, 'observations'),
            self.state.slice_array(val_states_arr, 'bias')
        ))
        mask_val = self.state.get_mask(val_states_arr).astype('bool')
        feature_mask_val = self.state.get_feature_mask(mask_val)
        assert(X_val.shape[0] == feature_mask_val.shape[0])
        assert(X_val.shape[1] == feature_mask_val.shape[1] + 1)
        y_val = val_states_labels

        return self._fit_val(
            X, mask, y,
            X_val, mask_val, y_val,
            num_workers)

    @abc.abstractmethod
    def _fit(self, X, mask, y, num_workers=1, fit_intercept=False):
        """
        Parameters
        ----------
        X: (N, D) ndarray of float
        mask: (N, F) ndarray of bool
        y: (N,) ndarray of int
        num_workers: int

        Returns
        -------
        best_score: float
            Best score obtained during the parameter search.
        """
        pass

    def predict_proba(self, state_vectors):
        """
        Return classifier multi-class confidences if has_been_fit,
        or random confidences otherwise.
        """
        N = 1 if state_vectors.ndim == 1 else state_vectors.shape[0]
        if self.has_been_fit:
            values = np.hstack((
                self.state.slice_array(state_vectors, 'observations'),
                self.state.slice_array(state_vectors, 'bias')
            ))
            mask = self.state.slice_array(state_vectors, 'mask').astype(bool)
            scores = self._predict_proba(values, mask)
        else:
            r = np.random.rand(N, self.num_classes)
            scores = r / r.sum()
        if N == 1:
            scores = scores.flatten()
        return scores

    @abc.abstractmethod
    def _predict_proba(self, X, mask=None):
        """
        Mask-informed prediction.
        """
        pass


class LogisticClassifier(PredictorClassifier):
    param_grid = [{
        'C': [.001, .1, 10, 100],
    }]

    def _fit(self, X, mask, y, num_workers=1, fit_intercept=False):
        """
        Tune parameters using cross-validation.
        """
        clf = sklearn.linear_model.LogisticRegression(
            fit_intercept=fit_intercept, class_weight='auto', dual=False, penalty='l2')
        kfold = KFold(y.shape[0], 2)
        clf = GridSearchCV(
            clf, self.param_grid, cv=kfold,
            verbose=0, n_jobs=num_workers, pre_dispatch=num_workers)
        clf.fit(X, y)
        print('Accuracy: {:.3f} achieved with {}'.format(clf.best_score_, clf.best_params_))
        self.clf = clf.best_estimator_
        self.has_been_fit = True
        return clf.best_score_

    def _fit_val(self, X, mask, y, X_val, mask_val, y_val, num_workers=1):
        """
        Tune parameters using validation data.
        """
        clfs = []
        scores = []
        params_list = list(sklearn.grid_search.ParameterGrid(self.param_grid))
        for params in params_list:
            clf = sklearn.linear_model.LogisticRegression(
                fit_intercept=False, class_weight='auto',
                dual=False, penalty='l2', **params)
            clf.fit(X, y)
            scores.append(clf.score(X_val, y_val))
            clfs.append(clf)
        best_ind = np.argmax(scores)
        print('Accuracy: {:.3f} achieved with {}'.format(scores[best_ind], params_list[best_ind]))
        self.clf = clfs[best_ind]
        self.has_been_fit = True
        return scores[best_ind]

    def _predict_proba(self, X, mask=None):
        return self.clf.predict_proba(X)


class SGDClassifier(PredictorClassifier):
    param_grid = [{
        'alpha': [.00001, .001, .1, 1],
    }]

    def _fit(self, X, y, num_workers=1, fit_intercept=False):
        clf = sklearn.linear_model.SGDClassifier(
            loss='log', penalty='l2', fit_intercept=fit_intercept,
            class_weight='auto', shuffle=False,
            warm_start=True, n_iter=int(np.ceil(10 ** 6 / X.shape[0])))

        fit_params = {}
        if hasattr(self, 'clf') and hasattr(self.clf, 'coef_'):
            fit_params = {
                'coef_init': self.clf.coef_,
                'intercept_init': self.clf.intercept_
            }
        kfold = KFold(y.shape[0], 2)
        clf = GridSearchCV(
            clf, self.param_grid, cv=kfold,
            fit_params=fit_params, verbose=0,
            n_jobs=num_workers, pre_dispatch=num_workers)
        clf.fit(X, y)
        print('Accuracy: {:.3f} achieved with {}'.format(
            clf.best_score_, clf.best_params_))
        self.clf = clf.best_estimator_
        self.has_been_fit = True
        return clf.best_score_

    def _fit_val(self, X, mask, y, X_val, mask_val, y_val, num_workers=1):
        """
        Tune parameters using validation data.
        """
        clfs = []
        scores = []
        params_list = list(sklearn.grid_search.ParameterGrid(self.param_grid))
        for params in params_list:
            # Get parameters of trained classifier if exists.
            # (It will always exist on the second time through this loop
            # at least.)
            if hasattr(self, 'clf') and hasattr(self.clf, 'coef_'):
                fit_params = {
                    'coef_init': self.clf.coef_,
                    'intercept_init': self.clf.intercept_
                }
            self.clf = sklearn.linear_model.SGDClassifier(
                loss='log', penalty='l2', fit_intercept=False,
                class_weight='auto', shuffle=False,
                warm_start=True, n_iter=int(np.ceil(10 ** 6 / X.shape[0])))
            self.clf.fit(X, y, **fit_params)
            scores.append(self.clf.score(X_val, y_val))
            clfs.append(self.clf)
        best_ind = np.argmax(scores)
        print('Accuracy: {:.3f} achieved with {}'.format(scores[best_ind], params_list[best_ind]))
        self.clf = clfs[best_ind]
        self.has_been_fit = True
        return scores[best_ind]

    def _predict_proba(self, X, mask=None):
        return self.clf.predict_proba(X)


class GaussianNBClassifier(PredictorClassifier):
    def plot_weights(self, filename=None):
        if hasattr(self.clf, 'theta_'):
            return tc.util.plot_weights(
                self.clf.theta_, xlabel='weights on features',
                ylabel='Classes', filename=filename)
        else:
            return None

    def _fit(self, X, mask, y, num_workers=1, fit_intercept=False):
        return self._fit_val(X, mask, y, X, mask, y, num_workers)

    def _fit_val(self, X, mask, y, X_val, mask_val, y_val, num_workers=1):
        self.clf = tc.GaussianNB().fit(X, y, mask)
        score = sklearn.metrics.accuracy_score(self.clf.predict(X_val, mask_val), y_val)
        print('Accuracy: {:.3f}'.format(score))
        self.has_been_fit = True
        return score

    def _predict_proba(self, X, mask=None):
        return self.clf.predict_proba(X, mask)
