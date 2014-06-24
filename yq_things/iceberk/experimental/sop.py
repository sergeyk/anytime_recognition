'''soap implements the second-order pooling idea.
@author: jiayq
'''

from iceberk import mathutil
import numpy as np
import scipy.linalg

class SecondOrderPooler(object):
    """wraps some of the common second-order pooling operations. Note that they
    are different from the general iceberk.pipeline.Pooler since they don't
    provide the pooling interface, so don't directly use them in your
    convolutional layers. Instead, use the 'method' keyword in the m
    """
    def __init__(self, reg = 1e-8, use_cov = False):
        # define some cache to make computation faster
        self._cache_triu_id = None
        self._cache_triu_size = -1
        self._cache_gram_matrix = None
        self._reg = reg
        self._use_cov = use_cov
    
    def _update_cache(self, X):
        """Updates the cache if necessary, based on the shape of the input X
        """
        if X.shape[1] != self._cache_triu_size:
            self._cache_triu_size = X.shape[1]
            self._cache_triu_id = np.triu_indices(X.shape[1])
            if not self._use_cov:
                self._cache_gram_matrix = np.empty((X.shape[1], X.shape[1]))

    def soap(self, X, out = None):
        """Performs second-order average pooling on a n*k matrix X. Returns a
        k*(k+1)/2 vector that is the upper triangular part of the average pooled
        region.
        """
        X = np.ascontiguousarray(X, dtype = np.float64)
        self._update_cache(X)
        if self._use_cov:
            self._cache_gram_matrix= np.cov(X, rowvar=0)
        else:
            mathutil.dot(X.T, X, out = self._cache_gram_matrix)
            self._cache_gram_matrix /= X.shape[0]
        if out is None:
            out = np.empty(X.shape[1] * (X.shape[1] + 1) / 2)
        out[:] = self._cache_gram_matrix[self._cache_triu_id[0],\
                                         self._cache_triu_id[1]]
        return out

    def somp(self, X, out = None):
        """Performs second-order max pooling on a n*k matrix X. Returns a
        k*(k+1)/2 vector that is the upper triangular part of the average pooled
        region.
        """
        X = np.ascontiguousarray(X, dtype = np.float64)
        self._update_cache(X)
        if out is None:
            out = np.empty(X.shape[1] * (X.shape[1] + 1) / 2)
        out[:] = - np.Inf
        for i in range(X.shape[0]):
            self._cache_gram_matrix = X[i][:, np.newaxis] * X[i]
            np.maximum(out, 
                       self._cache_gram_matrix[self._cache_triu_indices[0],\
                                               self._cache_triu_indices[1]],
                       out = out)
        return out

    def log_soap(self, X, out = None):
        """Performs matrix_log(second order average pooling)
        """
        X = np.ascontiguousarray(X, dtype = np.float64)
        self._update_cache(X)
        if self._use_cov:
            self._cache_gram_matrix= np.cov(X, rowvar=0)
        else:
            mathutil.dot(X.T, X, out = self._cache_gram_matrix)
            self._cache_gram_matrix /= X.shape[0]
        self._cache_gram_matrix.flat[::X.shape[1]+1] += self._reg
        # compute the matrix log
        logmat = scipy.linalg.logm(self._cache_gram_matrix)
        if out is None:
            out = np.empty(X.shape[1] * (X.shape[1] + 1) / 2)
        out[:] = logmat[self._cache_triu_id[0], self._cache_triu_id[1]]
        return out