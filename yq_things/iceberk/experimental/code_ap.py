"""
This code tries to do affinity propagation based on a set of input features
"""
from iceberk import mathutil, mpi
import logging
import numpy as np
from sklearn.cluster import AffinityPropagation

_AP_MAX_ITERATION = 20

def apcluster_k(feature, num_centers, corr = True, tol = 0):
    """perform the affinity propagation algorithm for the input codes.
    """
    logging.debug("ap: preparing similarity matrix")
    covmat = mathutil.mpi_cov(feature)
    std = np.diag(covmat)
    # normalize
    std = np.sqrt(std**2 + 0.01)
    if corr:
        # compute correlation. If corr is False, we will use the covariance
        # directly
        covmat /= std
        covmat /= std[:, np.newaxis]
    # compute the similarity matrix
    norm = np.diag(covmat) / 2
    covmat -= norm
    covmat -= norm[:, np.newaxis]
    # add a small noise to covmat
    noise = (covmat + np.finfo(np.float64).eps) * \
            np.random.rand(covmat.shape[0], covmat.shape[1])
    mpi.COMM.Bcast(noise)
    covmat += noise
    # The remaining part can just be carried out on root
    if mpi.is_root():
        # set preference
        pmax = covmat.max()
        #af = AffinityPropagation().fit(covmat, pmax)
        #num_max = len(af.cluster_centers_indices_)
        # in fact, num_max would always be covmat.shape[0] so we don't really
        # run ap
        num_max = covmat.shape[0]
        logging.debug("ap: pmax = %s, num = %d" % (pmax, num_max))
        pmin = covmat.min()
        af = AffinityPropagation().fit(covmat, pmin)
        # num_min is the theoretical min, but the python code seem to raise bugs...
        num_min = len(af.cluster_centers_indices_)
        logging.debug("ap: pmin = %s, num = %d" % (pmin, num_min))
        
        if num_centers < num_min:
            logging.warning("num_centers too small, will return %d centers" % (num_min,))
            return af.cluster_centers_indices_, af.labels_, covmat
    
        if num_centers > num_max:
            logging.warning("num_centers too large, will return everything.")
            return np.arange(covmat.shape[0], dtype=np.int), \
                   np.arange(covmat.shape[0], dtype=np.int)
        
        logging.debug("ap: start affinity propagation")
        
        # We will simply use bisection search to find the right number of centroids.
        for i in range(_AP_MAX_ITERATION):
            pref = (pmax + pmin) / 2
            af = AffinityPropagation().fit(covmat, pref)
            num = len(af.cluster_centers_indices_)
            logging.debug("ap try %d: pref = %s, num = %s" % (i + 1, pref, num))
            if num >= num_centers - tol and num <= num_centers + tol:
                break
            elif num < num_centers:
                pmin = pref
                num_min = num
            else:
                pmax = pref
                num_max = num
    else:
        af = None
    mpi.barrier()
    af = mpi.COMM.bcast(af)
    return af.cluster_centers_indices_, af.labels_, covmat

# backup compatibility
code_af = apcluster_k
