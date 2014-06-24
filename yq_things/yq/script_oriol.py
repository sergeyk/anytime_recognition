"""
The main script to test the accuracy on sbow imagenet
"""

import cPickle as pickle
from iceberk import mpi, classifier, mathutil
import numpy as np
import logging
import os, sys
import time
import gflags

########
# Settings
########
FEATDIR = "/u/vis/x1/common/ILSVRC-2010/SBOW/"
gflags.DEFINE_float("reg", 0.01, "The reg term")
gflags.DEFINE_integer("minibatch", 10000, "The minibatch size")
gflags.DEFINE_bool("svm", False, "If set, run SVM")
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

callback = \
        [lambda wb: classifier.Evaluator.accuracy(Yval, 
                (np.dot(Xval, wb[0]) + wb[1]).argmax(1)),
         lambda wb: classifier.Evaluator.accuracy(Ytest, 
                (np.dot(Xtest, wb[0]) + wb[1]).argmax(1))]

logging.info("Performing classification")

if FLAGS.svm:
    # do svm
    target = classifier.to_one_of_k_coding(Ytrain, fill = -1)
    loss = classifier.Loss2.loss_hinge
else:
    target = Ytrain.astype(np.int)
    loss = classifier.Loss2.loss_multiclass_logistic_yvector

solver = classifier.SolverStochastic(FLAGS.reg,
        loss,
        classifier.Reg.reg_l2,
        args = {'mode': 'adagrad', 'base_lr': 1e-7, 'minibatch': FLAGS.minibatch,
                'num_iter': 1000, 'callback': callback})
sampler = mathutil.NdarraySampler((Xtrain, target, None))
w,b = solver.solve(sampler, None, K = 1000)

pred = (np.dot(Xtrain, w) + b).argmax(1)
accu_train = classifier.Evaluator.accuracy(Ytrain, pred)
logging.info("Reg %f, train accu %f" % \
            (FLAGS.reg, accu_train))
if FLAGS.hier:
    mpi.root_pickle((w, b, FLAGS.reg, accu_train),
                    __file__ + str(FLAGS.reg) + ".hier.pickle")
elif FLAGS.svm:
    mpi.root_pickle((w, b, FLAGS.reg, accu_train),
                    __file__ + str(FLAGS.reg) + ".svm.pickle")
else:
    mpi.root_pickle((w, b, FLAGS.reg, accu_train),
                    __file__ + str(FLAGS.reg) + ".pickle")
