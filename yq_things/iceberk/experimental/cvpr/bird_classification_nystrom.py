'''
This script uses the iceberk pipeline to perform a cifar classification demo
using parameter settings idential to Adam Coates' AISTATS paper (except for the
number of kmeans centers, which we set to 800 for speed considerations).

You need to specify "--root=/path/to/cifar-data" to run the code. For other
optional flags, run the script with --help or --helpshort.

@author: jiayq
'''

import cPickle as pickle
import cProfile
import gflags
import logging
from iceberk import mpi, visiondata, pipeline, classifier, datasets, mathutil
from iceberk.experimental import code_ap
import numpy as np
import os
import sys

gflags.DEFINE_integer("grid", 0, "")
gflags.DEFINE_string("method", "", "")
gflags.DEFINE_integer("fromdim", 0, "")
gflags.DEFINE_integer("todim", 0, "")
gflags.DEFINE_integer("svd", 0, "")

FLAGS = gflags.FLAGS
gflags.FLAGS(sys.argv)


mpi.root_log_level(logging.INFO)
ROOT = '/u/vis/farrell/datasets/CUB_200_2011/'
CROP = 1.5
MIRRORED = True
SUBSET = None
TARGET_SIZE = [128,128]

mpi.root_log_level(logging.DEBUG)

logging.info('Loading bird data...')
train_data = visiondata.CUBDataset(ROOT, True, crop = CROP, subset = SUBSET,
                                  target_size = TARGET_SIZE, prefetch = True)
test_data = visiondata.CUBDataset(ROOT, False, crop = CROP, subset = SUBSET,
                                 target_size = TARGET_SIZE, prefetch = True)

conv = pipeline.ConvLayer([
        pipeline.PatchExtractor([6, 6], 1), # extracts patches
        pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
        pipeline.LinearEncoder({},
                trainer = pipeline.ZcaTrainer({'reg': 0.1})),
        pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': False},
                trainer = pipeline.NormalizedKmeansTrainer(
                     {'k': FLAGS.fromdim, 'max_iter':100})),
        pipeline.SpatialPooler({'grid': (FLAGS.grid, FLAGS.grid), 'method': FLAGS.method}) # average pool
        ])
logging.info("Total images: %d " % train_data.size_total())
print "local images: ", train_data.size()

logging.info('Training the pipeline...')
conv.train(train_data, 400000, exhaustive = True)

if MIRRORED:
    train_data = datasets.MirrorSet(train_data)
logging.info('Extracting features...')
Xtrain = conv.process_dataset(train_data, as_2d = False)
Ytrain = train_data.labels().astype(np.int)
Xtest = conv.process_dataset(test_data, as_2d = False)
Ytest = test_data.labels().astype(np.int)

# before we do feature computation, try to do dimensionality reduction
Xtrain.resize(np.prod(Xtrain.shape[:-1]), Xtrain.shape[-1])
Xtest.resize(np.prod(Xtest.shape[:-1]), Xtest.shape[-1])

m, std = classifier.feature_meanstd(Xtrain, 0.01)
Xtrain -= m
Xtrain /= std
Xtest -= m
Xtest /= std

covmat = mathutil.mpi_cov(Xtrain)

current_dim = FLAGS.todim
if FLAGS.svd == 1:
    eigval, eigvec = np.linalg.eigh(covmat)
# hack to run only one dim
while current_dim >= FLAGS.todim:
    if current_dim < FLAGS.fromdim:
        if FLAGS.svd == 1:
            # directly do dimensionality reduction
            U = eigvec[:, -current_dim:]
            Xtrain_red = np.dot(Xtrain, U)
            Xtest_red = np.dot(Xtest, U)
        else:
            # do subsampling
            temp = code_ap.code_af(Xtrain, current_dim)
            logging.info("selected %d dims" % len(temp[0]))
            sel = temp[0]
            sel = mpi.COMM.bcast(sel)
            Cpred = covmat[sel]
            Csel = Cpred[:,sel]
            W = np.linalg.solve(Csel, Cpred)
            # perform svd
            U, D, _ = np.linalg.svd(W, full_matrices = 0)
            U *= D
            Xtrain_red = np.dot(Xtrain[:, sel], U)
            Xtest_red = np.dot(Xtest[:, sel], U)
        Xtrain_red.resize(Ytrain.shape[0], Xtrain_red.size / Ytrain.shape[0])
        Xtest_red.resize(Ytest.shape[0], Xtest_red.size / Ytest.shape[0])
    else:
        Xtrain_red = Xtrain.copy()
        Xtest_red = Xtest.copy()
        Xtrain_red.resize(Ytrain.shape[0], Xtrain_red.size / Ytrain.shape[0])
        Xtest_red.resize(Ytest.shape[0], Xtest_red.size / Ytest.shape[0])
        
    w, b = classifier.l2svm_onevsall(Xtrain_red, Ytrain, 0.005,
                                     fminargs={'disp': 0, 'maxfun': 1000})
    accu_train = classifier.Evaluator.accuracy(Ytrain, np.dot(Xtrain_red, w) + b)
    accu_test = classifier.Evaluator.accuracy(Ytest, np.dot(Xtest_red, w) + b)
    logging.info('%d - %d, Training accuracy: %f' % (FLAGS.fromdim, current_dim, accu_train))
    logging.info('%d - %d, Testing accuracy: %f' % (FLAGS.fromdim, current_dim, accu_test))
    current_dim /= 2

