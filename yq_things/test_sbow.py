import cPickle as pickle
from iceberk import mpi, classifier, mathutil
from iceberk.experimental.meu_logistic_loss import loss_meu_logistic
import numpy as np
import logging
import os
from numpy.core.umath_tests import inner1d


########
# Settings
########
FEATDIR = '/tscratch/tmp/sergeyk/imagenet-sbow/'
RESULTSDIR = '/u/vis/x1/sergeyk/imagenet-sbow/'
LABELS = '/u/vis/x1/jiayq/ILSVRC/{}_predict/labels_ascii_sorted.npy'

########
# Main script
########
# load the classifier weights
wb = pickle.load(open('/u/vis/jiayq/codes/python/imagenet_exp/script_train_imagenet_sbow_stochastic.py0.0001.pickle'))

np.random.seed(42 + mpi.RANK)
mpi.root_log_level(level=logging.DEBUG)
for s in ['val', 'test', 'train']:
    logging.info("Loading data...")
    if s == 'train':
        print('Train accuracy is claimed to be: {:.3f}'.format(wb[-2]))
        # train is multiple matrices
        X = mpi.load_matrix_multi(os.path.join(FEATDIR, s, 'X{}'.format(s)))
    else:
        X = mpi.load_matrix(os.path.join(FEATDIR, s, 'X{}'.format(s)))
    X.resize(X.shape[0], np.prod(X.shape[1:]))
    # normalize to unit length
    X /= np.sqrt(inner1d(X, X)[:, np.newaxis] + 1e-8) / X.shape[1]
    Y = mpi.load_matrix(LABELS.format(s))
    print(X.shape)
    print(Y.shape)
    logging.info("Evaluating...")
    prob = np.dot(X, wb[0]) + wb[1]
    print('Accuracy on {}: {:.3f}'.format(s, classifier.Evaluator.accuracy(Y, prob.argmax(1))))
    np.save(os.path.join(RESULTSDIR, '{}_prob.npy'.format(s)), prob)
