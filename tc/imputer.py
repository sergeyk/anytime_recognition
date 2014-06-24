import numpy as np
import abc
import tc


class Imputer(object):
    def __init__(self, action_dims):
        self.state = tc.TimelyState(action_dims)
        self.has_been_fit = False

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, instances):
        """
        Parameters
        ----------
        instances: (N, D) ndarray
        """
        pass

    @abc.abstractmethod
    def impute(self, states):
        """
        Impute unobserved values in the array of states.

        Parameters
        ----------
        states: (N, S) or (S,) ndarray

        Returns
        -------
        states_imputed: (N, S) or (S,) ndarray
        """
        pass


class MeanImputer(Imputer):
    def fit(self, instances):
        self.mean = instances.mean(0)
        self.has_been_fit = True
        return self

    def impute(self, states):
        if states.ndim == 1:
            states = states[np.newaxis, :]

        N = states.shape[0]
        mask = self.state.slice_array(states, 'mask').astype(bool)
        feature_mask = self.state.get_feature_mask(mask)

        states_imputed = states.copy()
        observations = self.state.slice_array(states_imputed, 'observations')
        means = np.tile(self.mean, (N, 1))
        observations[feature_mask] = means[feature_mask]

        if N == 1:
            return states_imputed.flatten()
        return states_imputed


class GaussianImputer(Imputer):
    def fit(self, instances):
        self.S = np.cov(instances.T)
        self.mean = instances.mean(0)
        self.has_been_fit = True
        return self

    def impute(self, states):
        S = self.S

        if states.ndim == 1:
            states = states[np.newaxis, :]

        N = states.shape[0]
        mask = self.state.slice_array(states, 'mask').astype(bool)
        feature_mask = self.state.get_feature_mask(mask)

        # first, mean impute
        states_imputed = states.copy()
        Xm = self.state.slice_array(states_imputed, 'observations')
        means = np.tile(self.mean, (N, 1))
        Xm[feature_mask] = means[feature_mask]

        # now condition on observed values, if any
        for i in xrange(N):
            obs_ind = ~feature_mask[i, :]

            # if no features observed or all features observed, don't act
            if obs_ind.sum() == 0 or (~obs_ind).sum() == 0:
                continue

            # otherwise, do the dirty
            A = S[np.ix_(obs_ind, obs_ind)]
            C_T = S[np.ix_(~obs_ind, obs_ind)]
            ctainv = np.dot(C_T, np.linalg.pinv(A))
            mean = np.dot(ctainv, Xm[i, obs_ind])
            Xm[i, ~obs_ind] = mean

        if N == 1:
            return states_imputed.flatten()
        return states_imputed
