"""
The main script to test the accuracy on birds
"""

import cPickle as pickle
from iceberk import mpi, classifier, mathutil
from iceberk.experimental.meu_logistic_loss import loss_meu_logistic
import numpy as np
import logging
import os, sys
from sklearn.cross_validation import StratifiedKFold
import time
import gflags

########
# Settings
########
FEATDIR = "/tscratch/tmp/jiayq/imagenet-sbow/"
gflags.DEFINE_bool("load", False, "If set, load the data and then stop.")
gflags.DEFINE_float("reg", 0.01, "The reg term")
gflags.DEFINE_integer("minibatch", 100000, "The minibatch size")
gflags.DEFINE_bool("svm", False, "If set, run SVM")
gflags.DEFINE_bool("hier", False, "If set, use hierarchical loss")
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

########
# Main script
########
if mpi.SIZE > 1:
    raise RuntimeError, "This script runs on single machines only."

np.random.seed(42 + mpi.RANK)
mpi.root_log_level(level=logging.DEBUG)
logging.info("Loading data...")
Xtrain = mpi.load_matrix_multi(os.path.join(FEATDIR,'train', 'Xtrain'))
Ytrain = mpi.load_matrix(os.path.join(FEATDIR,'train', 'Ytrain.npy'))
Xtrain.resize(Xtrain.shape[0], np.prod(Xtrain.shape[1:]))

# normalize to unit length
for i in range(Xtrain.shape[0]):
    Xtrain[i] /= np.sqrt(np.dot(Xtrain[i],Xtrain[i]) + 1e-8) / Xtrain.shape[1]

logging.info("Performing classification")
target = classifier.to_one_of_k_coding(Ytrain, fill = 0)

# stochastic lbfgs - we play a little trick by using all the training data to do initial lbfgs
solver = classifier.SolverStochastic(FLAGS.reg,
        classifier.Loss2.loss_multiclass_logistic,
        classifier.Reg.reg_l2,
        args = {'mode': 'lbfgs', 'minibatch': FLAGS.minibatch, 'num_iter': 20},
        fminargs = {'maxfun': 20, 'disp': 0})
sampler = mathutil.NdarraySampler((Xtrain, target, None))
w,b = solver.solve(sampler)
logging.info("Stochastic LBFGS done.")

skf = StratifiedKFold(Ytrain, k = 10)
skf_results = []
for train_index, test_index in skf:
    param_init = (w,b)
    solver = classifier.SolverStochastic(FLAGS.reg,
            classifier.Loss2.loss_multiclass_logistic,
            classifier.Reg.reg_l2,
            args = {'mode': 'adagrad', 'base_lr': 1e-4, 'minibatch': FLAGS.minibatch,
                    'num_iter': 1000})
    del target
    target = classifier.to_one_of_k_coding(Ytrain[train_index], fill = 0)
    sampler = mathutil.NdarraySampler((Xtrain[train_index], target, None))
    ww, bb = solver.solve(sampler, param_init)
    skf_results.append((ww,bb,train_index,test_index))

mpi.root_pickle(skf_results,
                    __file__ + str(FLAGS.reg) + ".pickle")
