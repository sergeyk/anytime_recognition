"""
The main script to test the accuracy on birds
"""

import cPickle as pickle
from iceberk import mpi, classifier, mathutil
from iceberk.experimental.meu_logistic_loss import loss_meu_logistic
import numpy as np
import logging
import os, sys
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

if FLAGS.load:
    logging.info("Loading done.")
    sys.exit(0)

logging.info("Performing classification")

if FLAGS.hier:
    # todo: obtain the infogain
    from birdmix import tax
    graph = tax.get_imagenet_taxonomy(1000)
    leaves = [n for n in graph.nodes() if len(graph.successors(n)) == 0]
    leaves.sort()
    leaf2id = dict((n,i) for i, n in enumerate(leaves))
    infogain = tax.pairwise_info_gain(graph)
    # convert to mat
    igmat = np.zeros((1000,1000))
    for key in infogain:
        igmat[leaf2id[key[0]], leaf2id[key[1]]] = infogain[key]
    np.exp(igmat, igmat)
    # normalize
    igmat /= igmat.sum(1)[:, np.newaxis]
    target = np.ascontiguousarray(igmat[Ytrain.astype(np.int)])
    loss = loss_meu_logistic
elif FLAGS.svm:
    # do svm
    target = classifier.to_one_of_k_coding(Ytrain, fill = -1)
    loss = classifier.Loss2.loss_hinge
else:
    target = Ytrain.astype(np.int)
    loss = classifier.Loss2.loss_multiclass_logistic_yvector

# stochastic lbfgs
use_lbfgs = True
if use_lbfgs:
    solver = classifier.SolverStochastic(FLAGS.reg,
            loss,
            classifier.Reg.reg_l2,
            args = {'mode': 'lbfgs', 'minibatch': FLAGS.minibatch, 'num_iter': 10, 'callback': callback},
            fminargs = {'maxfun': 10, 'disp': 0})
    sampler = mathutil.NdarraySampler((Xtrain, target, None))
    w,b = solver.solve(sampler, K = 1000)
    logging.info("Stochastic LBFGS done.")

# adagrad
if use_lbfgs:
    param_init = (w,b)
else:
    param_init = None

solver = classifier.SolverStochastic(FLAGS.reg,
        loss,
        classifier.Reg.reg_l2,
        args = {'mode': 'adagrad', 'base_lr': 1e-3, 'minibatch': FLAGS.minibatch,
                'num_iter': 1000, 'callback': callback})
sampler = mathutil.NdarraySampler((Xtrain, target, None))
w,b = solver.solve(sampler, param_init, K = 1000)

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
