import logging
import numpy as np
from iceberk import mathutil, mpi, pipeline

def conditional_covariance(covmat, idx, reg=1e-8):
    """This function returns the conditional variance of all the features, given
    a set of feature indices provided in idx as a numpy array. The returned
    covariance matrix would have the same size as the original covariance matrix
    (For those features that are selected, their conditional covariance would
    simply be zero). The equation is:
    C_cond = C - C[:,idx] * (C[idx,idx])^-1 * C[idx,:]
    """
    covmat_given = np.ascontiguousarray(covmat[idx][:, idx])
    covmat_inter = np.ascontiguousarray(covmat[idx])
    # compute the inverse
    covmat_given[::len(idx)+1] += reg
    inv_covmat_given = np.linalg.pinv(covmat_given)
    covmat_cond = covmat.copy()
    covmat_cond -= np.dot(covmat_inter.T, 
                          np.dot(inv_covmat_given, covmat_inter))
    return covmat_cond


def max_variance_feature_selection(covmat, num_features, reg=1e-8,
                                   refresh_covmat = 10):
    """I still don't quite get the math behind this feature selection yet, but
    basically, what I do here is to select the feature with the largest 
    variance, and then remove its impact from the remaining features (using
    conditional probability), and select the next feature with the largest
    conditional variance, and so on.
    """
    if num_features <= 0:
        raise ValueError, "num_features should be positive"
    indices = np.zeros(num_features, dtype=int)
    covmat_cond = covmat.copy()
    for i in range(num_features):
        best_id = np.argmax(np.diag(covmat_cond))
        indices[i] = best_id
        if i > 0 and i % refresh_covmat == 0:
            # we need to refresh the covariance matrix by completely recomputing
            covmat_cond = conditional_covariance(covmat, indices[:i+1], reg)
        else:
            covmat_cond = conditional_covariance(covmat_cond, 
                                                 np.array([best_id]),
                                                 reg)
    return indices

def prune_conv(conv, dataset, num_patches, num_features):
    if not isinstance(conv[-1], pipeline.Pooler):
        raise TypeError, "The last layer should be a pooler."
    if not isinstance(conv[-2], pipeline.FeatureEncoder):
        raise TypeError, "The second last layer should be an encoder."
    logging.debug('Randomly sampling pooled features...')
    features = conv.sample(dataset, num_patches, True)
    if features.shape[1] != conv[-2].dictionary.shape[0]:
        raise ValueError, "Huh, I can't figure out the encoding method.\n"\
                "Feature shape: %d, dictionary size: %d" % \
                (features.shape[1], conv[-2].dictionary.shape[0])
    logging.debug('Perform feature selection...')
    covmat = mathutil.mpi_cov(features)
    if mpi.is_root():
        selected_idx = max_variance_feature_selection(covmat, num_features)
    else:
        selected_idx = None
    selected_idx = mpi.COMM.bcast(selected_idx)
    conv[-2].dictionary = conv[-2].dictionary[selected_idx]
    return covmat
