import numpy as np
import fastcluster
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import fcluster
import tc

def training_predict(X, K):
    """
    Get unique masks and cluster indices on the training set.

    Parameters
    ----------
    X : (N, F) ndarray of boolean

    Returns
    -------
    umasks : (UK, F) ndarray of bool

    cluster_ind : (N,) ndarray of int
        Each cluster ind is [0, K'), with K' <= K,
        or [0, UK) if K == -1 or K >= UK.
    """
    umasks = tc.mask_distribution.get_unique_masks(X)
    UK = umasks.shape[0]
    if K < 0 or K >= UK:
        cluster_ind = np.zeros(X.shape[0], dtype=int)
        for i in range(1, UK):
            cluster_ind[(X == umasks[i]).all(1)] = i
    else:
        Z = fastcluster.linkage(X, method='single', metric='hamming')
        cluster_ind = fcluster(Z, K, criterion='maxclust') - 1
    return umasks, cluster_ind


class MaskClustering(object):
    """
    Cluster rows of Boolean ndarrays by Hamming distance.

    Parameters
    ----------
    K : int
        If -1, all unique masks are found.
    """
    def __init__(self, K):
        self.K = K

    def umask_for_cluster(self, cluster_ind):
        return self.umasks[self.umask_to_cluster_map.index(cluster_ind)]

    def fit(self, X):
        self.umasks, cluster_ind = training_predict(X, self.K)
        self.umask_to_cluster_map = []
        for umask in self.umasks:
            # get the cluster_ind of the masks that correspond to this
            # unique mask, and make sure they are all equal
            inds = cluster_ind[(X == umask).all(1)]
            assert(np.all([ind == inds[0] for ind in inds]))
            self.umask_to_cluster_map.append(inds[0])
        return self

    def predict(self, X):
        if not (hasattr(self, 'umasks') and hasattr(self, 'umask_to_cluster_map')):
            raise Exception('Must call fit() before calling predict()')
        N = X.shape[0]
        cluster_ind = np.zeros(N)
        # TODO: deal with X being a vector here: must convert to 2d array
        dists = pairwise_distances(X, self.umasks, metric='hamming')
        cluster_ind = [self.umask_to_cluster_map[dists[i].argmin()] for i in range(N)]
        return np.array(cluster_ind)
