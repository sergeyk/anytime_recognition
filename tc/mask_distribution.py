import numpy as np
from sklearn.metrics import pairwise_distances


def get_unique_masks(masks):
    """
    Return ndarray of unique masks in an ndarray of boolean masks.

    Parameters
    ----------
    masks: (N, F) ndarray of bool

    Returns
    -------
    umasks: (U, F) ndarray of bool
    """
    rows = [row.tostring() for row in masks]
    urows = np.unique(rows)
    U = len(urows)
    umasks = np.fromstring(urows, dtype='bool').reshape(U, -1)
    return umasks


def sample_feasible_mask(ds):
    mask = np.ones(len(ds.actions), dtype=bool)
    costs = ds.action_costs.copy()
    while True:
        feas_inds = np.flatnonzero(mask & (costs <= ds.max_budget))
        if len(feas_inds) == 0:
            return mask
        ind = feas_inds[np.random.randint(len(feas_inds))]
        mask[ind] = False
        costs += ds.action_costs[ind]


class MaskDistribution(object):
    """
    Represent distribution over binary masks that can be updated with
    array of TimelyState.

    Parameters
    ----------
    max_masks: int, optional [None]
        If given, specifies the maximum number of unique masks to
        maintain.

    Properties
    ----------
    umasks: (U, F) ndarray of bool
        Where U is the number of unique masks that have ever been seen, and
        F is the mask feature dimensions.
    counts: (U,) ndarray of float
        Distribution over the masks: sums to 1, each number in [0,1].
    """
    def __init__(self, max_masks=None):
        self.umasks = None
        self.counts = None
        self.max_masks = max_masks

    def update(self, masks):
        """
        Update the distribution with new masks.
        At the end, masks are re-sorted by frequency (descending).

        Parameters
        ----------
        masks: (N, F) ndarray of bool
        threshold: float, optional [None]
            Only include masks that occur at least threshold fraction
            of the time.
        """
        if masks.ndim == 1:
            masks = masks[np.newaxis, ]
        umasks = get_unique_masks(masks)
        counts = np.array([(masks == umask).all(1).sum() for umask in umasks])

        if self.umasks is None:
            self.umasks = umasks
            self.counts = counts
        else:
            new_masks = []
            new_counts = []
            for umask, count in zip(umasks, counts):
                already_present_ind = np.flatnonzero((self.umasks == umask).all(1))
                if len(already_present_ind) > 0:
                    self.counts[already_present_ind[0]] += count
                else:
                    new_masks.append(umask)
                    new_counts.append(count)
            if len(new_masks) > 0:
                self.umasks = np.vstack((self.umasks, np.array(new_masks)))
                self.counts = np.hstack((self.counts, np.array(new_counts)))

        # re-sort
        ind = np.argsort(-self.dist)
        self.umasks = self.umasks[ind]
        self.counts = self.counts[ind]

        # get rid of excess masks
        if self.max_masks is not None:
            self.umasks = self.umasks[:self.max_masks]
            self.counts = self.counts[:self.max_masks]

        return self

    @property
    def dist(self):
        """
        Return distribution (normalized counts) over masks.
        """
        return self.counts.astype(float) / self.counts.sum()

    def sample(self, N, shuffled=True):
        """
        Sample N masks from the distribution.
        Sampling is done with the multinomial distribution.

        Parameters
        ----------
        N: int
        shuffled: bool, optional [True]
            If False, then masks will be returned in the order of self.umasks.

        Returns
        -------
        masks: (N, F) ndarray of bool
        """
        if self.umasks is None:
            raise Exception("Cannot sample from a distribution that has never been updated.")
        repeats = np.random.multinomial(N, self.dist)
        masks = np.repeat(self.umasks, repeats, axis=0)
        if shuffled:
            masks = masks[np.random.permutation(N)]
        return masks

    def predict_cluster(self, masks, K):
        """
        Match given masks to top K clusters.

        Parameters
        ----------
        masks: (N, F) ndarray of boolean

        Returns
        -------
        cluster_inds: (N,) ndarray of int
            Each cluster_ind is [0, K'), with K' = min(K, UK).
            If K == -1, K' = UK.
        """
        N = masks.shape[0]
        if masks.ndim == 1:
            masks = masks[np.newaxis, :]
        if K < 1:
            K = self.umasks.shape[0]
        dists = pairwise_distances(masks, self.umasks[:K], metric='hamming')
        cluster_inds = np.array([dists[i].argmin() for i in range(N)])
        return cluster_inds
