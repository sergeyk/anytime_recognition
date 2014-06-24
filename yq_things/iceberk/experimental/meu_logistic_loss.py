from iceberk import mathutil
import numpy as np
from numpy.core.umath_tests import inner1d

def loss_meu_logistic(Y, pred, weight, gpred, cache, **kwargs):
    """This loss function computes the maximum expected utility based loss
    where the input Y is the normalized utility values.
    """
    # compute the probability, normalize prediction to avoid overflowing
    if len(cache) == 0:
        cache.append(np.empty_like(pred))
    cache[0].resize(pred.shape)
    prob = cache[0]
    prob[:] = pred
    prob -= pred.max(axis=1)[:,np.newaxis]
    mathutil.exp(prob, out=prob)
    prob /= prob.sum(axis=1)[:, np.newaxis]
    # eu is the expected utility
    eu = inner1d(Y, prob)
    # numerical stability
    eu += np.finfo(np.float64).eps
    gpred[:] = Y * prob
    gpred /= - eu[:, np.newaxis]
    gpred += prob
    mathutil.log(eu, out=eu)
    return - eu.sum(), gpred
