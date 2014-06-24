import abc
import numpy as np
from numpy.random import rand, randint
import sklearn
from sklearn.linear_model import Ridge
import tc


class Policy(object):
    """
    A Policy must implement the select_action() and predict() methods, and will
    get calls to fit() and plot_weights().
    It's fine to return None for plot_weights().

    Parameters
    ----------
    ds: tc.DataSource
    """
    def __init__(self, ds):
        self.ds = ds
        self.F = len(self.ds.actions)
        self.state = tc.TimelyState(self.ds.action_dims)

    __metaclass__ = abc.ABCMeta

    def __repr__(self):
        return self.__class__.__name__

    def plot_weights(self, filename=None):
        return None

    def fit(self, states_arr, actions, scores, num_workers=1):
        """
        This is called in training.

        Parameters
        ----------
        states_arr: (N, D) ndarray of float
            The D-dimensional data to fit.
        actions: (N,) ndarray of int
            Labels for each data point, indicating which action was taken.
        scores: (N,) ndarray of float
            Values to regress to.
        num_workers: int
        """
        return -1

    def select_action(self, state_vector, epsilon=0):
        """
        Select action with max value.

        Parameters
        ----------
        state_vector: (S,) ndarray
        epsilon: non-negative float, optional [0]
            Value to use in epsilon-greedy action selection.

        Returns
        -------
        action_ind: int
            The index of the untaken action of max value, or -1.
        """
        if epsilon == 0 or rand() > epsilon:
            return self.predict(state_vector).argmax()
        else:
            return randint(self.F)

    def select_untaken_action(self, state_vector, epsilon=0):
        """
        Select untaken action with max value, or return -1 if no more actions.

        Parameters and Returns as in select_action().
        """
        untaken_inds = np.flatnonzero(self.state.slice_array(state_vector, 'mask'))
        if len(untaken_inds) == 0:
            return -1
        if epsilon == 0 or rand() > epsilon:
            action_values = self.predict(state_vector)
            ind = action_values[untaken_inds].argmax()
        else:
            action_values = np.zeros(self.F)
            ind = randint(len(untaken_inds))
        return untaken_inds[ind]

    @abc.abstractmethod
    def predict(self, states_arr):
        """
        Return scores for each state in the states_arr, which can be a vector
        or matrix.

        Parameters
        ----------
        states_arr: (N, D) or (D,) ndarray of float

        Returns
        -------
        scores: (N, F) or (F,) ndarray of float
        """
        pass

    def random_predict(self, states_arr):
        """
        Return random scores.

        Parameters and Returns as in predict().
        """
        return rand(self.F) if states_arr.ndim == 1 else rand(states_arr.shape[0], self.F)


class ManualOrderedPolicy(Policy):
    """
    Select actions in a pre-specified order.
    This is just for debugging.
    """
    def __init__(self, ds):
        super(ManualOrderedPolicy, self).__init__(ds)
        self.actions = np.arange(self.F)

    def __repr__(self):
        return "{}: {}".format(
            self.__class__.__name__, self.actions)

    def select_action(self, state_vector, epsilon=0):
        untaken_inds = np.flatnonzero(self.state.slice_array(state_vector, 'mask'))
        if len(untaken_inds) < 1:
            return -1
        if epsilon == 0 or rand() > epsilon:
            return self.actions[self.F - len(untaken_inds)]
        else:
            return randint(self.F)

    def predict(self, states_arr):
        # can't be bothered to return the actual distribution right now
        return self.random_predict(states_arr)


class ManualOrthantsPolicy(Policy):
    """
    Select actions in the optimal way.
    """
    def select_action(self, state_vector, epsilon=0):
        # no epsilon, ever
        return self.predict(state_vector).argmax()

    def predict(self, states_arr):
        def get_dist(state_vector):
            scores = np.zeros(self.F)

            # The 'd' actions are first, so take them until they are all taken
            untaken_inds = np.flatnonzero(self.state.slice_array(state_vector, 'mask'))
            for d in range(self.ds.D):
                if d in untaken_inds:
                    scores[d] = 1
                    return scores

            # With all the 'd' observations, we can find the correct quadrant action
            # NOTE: this assumes that the cheap actions are first...
            d_obs = self.state.slice_array(state_vector, 'observations')[:self.ds.D]
            q = np.dot(self.ds.quadrant_indicators, d_obs).argmax()
            scores[self.ds.D + q] = 1
            return scores

        if states_arr.ndim == 1:
            return get_dist(states_arr)
        else:
            return np.array([get_dist(state_vector) for state_vector in states_arr])


class RandomPolicy(Policy):
    """
    Select untaken actions in a random order.
    """
    def select_action(self, state_vector, epsilon=0):
        return self.select_untaken_action(state_vector, epsilon)

    def predict(self, states_arr):
        return self.random_predict(states_arr)


class LinearPolicy(Policy):
    """
    Predictor object for F separate linear regressions, one for each action.
    """
    def __init__(self, ds):
        super(LinearPolicy, self).__init__(ds)
        self.predictors = [None for f in range(self.F)]
        self.has_been_fit = False

    def __repr__(self):
        clf = self.predictors[0] if hasattr(self, 'predictors') else 'not fit yet'
        return '{}: {}'.format(
            self.__class__.__name__, clf)

    def predict(self, states_arr):
        """
        Predict F values for the given data points using F separate regressions.
        If regressions have not been trained yet, return random scores.
        """
        if not (hasattr(self, 'has_been_fit') and self.has_been_fit):
            return self.random_predict(states_arr)

        # TODO: can speed this up by having two cases instead of flattening
        N = 1 if states_arr.ndim == 1 else states_arr.shape[0]
        scores = np.zeros((N, self.F))
        for i, regr in enumerate(self.predictors):
            # Individual regressions may not be trained if fit() didn't see
            # their action indices.
            try:
                scores[:, i] = regr.predict(states_arr)
            except:
                scores[:, i] = rand()
        if N == 1:
            scores = scores.flatten()
        return scores

    def fit_(self, states_arr, actions, scores):
        """
        Fit F separate predictors, such that predictor number i is only fit
        with data points whose label in a is i.
        """
        for action_ind in np.unique(actions):
            ind = np.flatnonzero(actions == action_ind)
            clf = Ridge(fit_intercept=False, solver='lsqr', alpha=1)
            try:
                clf.fit(states_arr[ind, :], scores[ind])
                self.predictors[action_ind] = clf
            except np.linalg.linalg.LinAlgError:
                print("Could not train due to LinAlgError!")
        self.has_been_fit = True

    def fit(self, states_arr, actions, scores, num_workers=1):
        # TODO: actually incorporate num_workers

        cv = sklearn.cross_validation.KFold(states_arr.shape[0], n_folds=3)
        mses = []
        for train, test in cv:
            self.fit_(states_arr[train, :], actions[train], scores[train])
            scores_pred = self.predict(states_arr[test, :])

            mse = []
            for action_ind in np.unique(actions):
                ind = np.flatnonzero(actions[test] == action_ind)
                mse.append(sklearn.metrics.mean_squared_error(scores_pred[ind, action_ind], scores[ind]))
            mses.append(np.mean(mse))

        mses = np.array(mses)
        print('MSE is {:.3f} +/- {:.3f}'.format(mses.mean(), mses.std()))
        self.fit_(states_arr, actions, scores)
        return mses.mean()

    def plot_weights(self, filename=None):
        """
        Plot policy weights.
        """
        if not self.has_been_fit:
            return None
        weights = []
        for r in self.predictors:
            if r is not None:
                weights.append(r.coef_)
            else:
                weights.append(np.zeros(self.state.S))
        try:
            fig = tc.util.plot_weights(
                np.array(weights), xlabel='weights on $\phi(s)$',
                ylabel='Actions', yticks=self.ds.actions,
                filename=filename)
            return fig
        except Exception as e:
            print(e)
            return None


class LinearUntakenPolicy(LinearPolicy):
    """
    As LinearPolicy, but cannot repeat actions.
    """
    def select_action(self, state, epsilon=0):
        return self.select_untaken_action(state, epsilon)


class StaticLinearPolicy(LinearPolicy):
    """
    The static policy does not use the observation features of the state.
    """
    def fit(self, states_arr, actions, scores, num_workers):
        states_arr = self.state.get_mask(states_arr, with_bias=True)
        return super(StaticLinearPolicy, self).fit(states_arr, actions, scores, num_workers)

    def predict(self, states_arr):
        states_arr = self.state.get_mask(states_arr, with_bias=True)
        return super(StaticLinearPolicy, self).predict(states_arr)


class StaticLinearUntakenPolicy(StaticLinearPolicy):
    """
    As StaticLinearPolicy, but cannot repeat actions.
    """
    def select_action(self, state, epsilon=0):
        return self.select_untaken_action(state, epsilon)
