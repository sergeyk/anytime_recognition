
import cPickle as pickle
import coclassification as coc
from iceberk import mpi, classifier
import numpy as np
import logging
import os, sys
import time
import gflags

########
# Settings
########
FEATDIR = "/tscratch/tmp/jiayq/imagenet-sbow/"
MODEL_NAME = 'script_train_imagenet_sbow_stochastic.py.flat.0.001.pickle'
NTRIAL = 10000
SETSIZE = 10
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

########
# Main script
########
np.random.seed(42 + mpi.RANK)
mpi.root_log_level(level=logging.DEBUG)
logging.info("Loading data...")
Xtrain = mpi.load_matrix_multi(os.path.join(FEATDIR,'train', 'Xtrain'))
Ytrain = mpi.load_matrix(os.path.join(FEATDIR,'train', 'Ytrain.npy'))
Xtrain.resize(Xtrain.shape[0], np.prod(Xtrain.shape[1:]))

Xval = mpi.load_matrix(os.path.join(FEATDIR, 'val', 'Xval'))
Yval = mpi.load_matrix(os.path.join(FEATDIR, 'val', 'Yval'))
Xval.resize(Xval.shape[0], np.prod(Xval.shape[1:]))

Xtest = mpi.load_matrix(os.path.join(FEATDIR, 'test', 'Xtest'))
Ytest = mpi.load_matrix(os.path.join(FEATDIR, 'test', 'Ytest'))
Xtest.resize(Xtest.shape[0], np.prod(Xtest.shape[1:]))

# normalize to unit length
for i in range(Xtrain.shape[0]):
    Xtrain[i] /= np.sqrt(np.dot(Xtrain[i],Xtrain[i]) + 1e-8) / Xtrain.shape[1]
for i in range(Xval.shape[0]):
    Xval[i] /= np.sqrt(np.dot(Xval[i],Xval[i]) + 1e-8) / Xval.shape[1]
for i in range(Xtest.shape[0]):
    Xtest[i] /= np.sqrt(np.dot(Xtest[i],Xtest[i]) + 1e-8) / Xtest.shape[1]

w, b = pickle.load(open(MODEL_NAME))[0]
pred_train = (np.dot(Xtrain, w) + b).argmax(1)
pred_val = (np.dot(Xval, w) + b).argmax(1)
pred_test = (np.dot(Xtest, w) + b).argmax(1)

confmat_train = classifier.Evaluator.confusion_table(Ytrain, pred_train)
confmat_val = classifier.Evaluator.confusion_table(Yval, pred_val)
concept = coc.FlatConcept(confmat_val, ('kneserney', 0.75))

def obtain_accuracy(concept, Ytest, pred_test, ntrial, setsize):
    # first, get the indices
    idx = []
    for i in range(concept.K):
        idx.append(np.flatnonzero(Ytest == i))
    accu = 0
    accu_count = 0
    accu_baseline = 0
    for i in range(ntrial):
        label = np.random.randint(concept.K)
        batch = idx[label]
        np.random.shuffle(batch)
        gamma, theta = concept.coclassify(pred_test[batch[:setsize]])
        accu += sum(theta.argmax(1) == label)
        accu_count += sum(concept.coclassify_flat_vote_baseline(pred_test[batch[:setsize]]) == label)
        accu_baseline += sum(pred_test[batch[:setsize]] == label)
    accu = accu / float(ntrial) / float(setsize)
    accu_count = accu_count / float(ntrial) / float(setsize)
    accu_baseline = accu_baseline / float(ntrial) / float(setsize)
    return accu, accu_count, accu_baseline
