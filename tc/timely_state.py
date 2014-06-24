import numpy as np
from tc.util import slice_array


class TimelyState(object):
    """
    Class abstracting what needs to be kept track of when running sequential
    classification.

    The state consists of:
    - F bits:       Mask of computed features.
                    NB: 1 if feature is unobserved.
    - D floats:     Feature observations.
    - 1 float:      Fraction of the budget spent.
    - 1 (constant): Bias.

    Parameters
    ----------
    action_dims: int sequence of length A
        Feature dimensions of each possible action.
    """
    def __init__(self, action_dims):
        # Figure out the feature bounds and initialize the vector.
        self.F = len(action_dims)
        self.D = np.sum(action_dims)
        feature_bounds = np.hstack((0, np.cumsum(action_dims, dtype=int)))
        self.feature_bounds = zip(feature_bounds[:-1], feature_bounds[1:])

        intervals = [
            ('mask', self.F),
            ('observations', self.D),
            ('cost', 1),
            ('bias', 1)
        ]
        names, sizes = zip(*intervals)
        bounds = np.hstack((0, np.cumsum(sizes)))
        self.bounds = dict(zip(names, zip(bounds[:-1], bounds[1:])))
        self.S = sum(sizes)

    def get_initial_state(self, N=1):
        """
        Return vector corresponding to initial state.
        """
        if N > 1:
            array = np.zeros(self.N, self.S)
        else:
            array = np.zeros(self.S)
        self.slice_array(array, 'mask')[:] = 1
        self.slice_array(array, 'bias')[:] = 1
        return array

    def slice_array(self, vector, name):
        """
        Return the part of the given state vector that corresponds to the given
        name.
        """
        return slice_array(vector, self.bounds, name)

    def get_mask(self, array, with_bias=False):
        """
        Return the observation masks and, optionally, the bias column as ndarray
        of int.

        Parameters
        ----------
        array: (N, S) or (S,) ndarray
            Of states.
        with_bias: bool
            If True, append a column of 1 at the end.

        Returns
        -------
        masks: (N, F') or (F',) ndarray of int
            Where F' = F + 1 if with_bias, and F otherwise.
        """
        masks = self.slice_array(array, 'mask')
        if with_bias:
            if masks.ndim == 1:
                masks = np.hstack((masks, 1))
            else:
                masks = np.hstack((masks, np.ones((masks.shape[0], 1))))
        return masks.astype(int)

    def get_state(self, instance, action_inds, cost):
        """
        Featurize the state.

        Parameters
        ----------
        instance: (D,) ndarray of float
        action_inds: list of int
            Actions that have been taken.
        cost: float
        """
        # Zero-impute the unobserved features.
        mask = np.ones(self.F)
        mask[action_inds] = 0
        feature_mask = self.get_feature_mask(mask == 1)
        observations = np.array(instance)
        observations[feature_mask] = 0

        vector = np.ones(self.S)
        self.slice_array(vector, 'mask')[:] = mask
        self.slice_array(vector, 'observations')[:] = observations
        self.slice_array(vector, 'cost')[:] = cost
        return vector

    def get_feature_mask(self, mask):
        """
        Parameters
        ----------
        mask: (N, F) or (F,) ndarray of bool
            True for unobserved features.

        Returns
        -------
        feature_mask: (N, D) or (D,) ndarray of bool
            True for unobserved features.
        """
        assert(mask.dtype == bool)
        if mask.ndim == 1:
            assert(mask.shape[0] == self.F)
            feature_mask = np.ones(self.D).astype(bool)
            for ind in np.flatnonzero(~mask):
                feature_mask[slice(*self.feature_bounds[ind])] = False
        else:
            assert(mask.shape[1] == self.F)
            N = mask.shape[0]
            feature_mask = np.ones((N, self.D)).astype(bool)
            for i in xrange(N):
                for ind in np.flatnonzero(~mask[i]):
                    feature_mask[i, slice(*self.feature_bounds[ind])] = False
        return feature_mask

    def get_states_from_mask(self, instances, mask, costs=None):
        """
        Parameters
        ----------
        instances: (N, D) ndarray
        mask: (N, F) ndarray of bool

        Returns
        -------
        states: (N, S) ndarray
        """
        N = instances.shape[0]
        feature_mask = self.get_feature_mask(mask)
        observations = instances.copy()
        observations[feature_mask] = 0
        if costs is None:
            costs = np.ones(N)

        states = np.ones((N, self.S))
        self.slice_array(states, 'mask')[:] = mask
        self.slice_array(states, 'observations')[:] = observations
        self.slice_array(states, 'cost')[:] = np.atleast_2d(costs).T
        return states
