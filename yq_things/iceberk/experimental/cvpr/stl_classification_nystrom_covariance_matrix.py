'''
This script uses the iceberk pipeline to perform a stl classification demo
using parameter settings idential to Adam Coates' AISTATS paper (except for the
number of kmeans centers, which we set to 800 for speed considerations).

You need to specify "--root=/path/to/stl-data" to run the code. For other
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

gflags.DEFINE_string("root", "",
                     "The root to the stl dataset (python format)")
gflags.RegisterValidator('root', lambda x: x != "",
                         message='--root must be provided.')

gflags.DEFINE_integer("grid", 0, "")
gflags.DEFINE_string("method", "", "")
gflags.DEFINE_integer("fromdim", 0, "")
gflags.DEFINE_integer("svd", 0, "")

FLAGS = gflags.FLAGS

def stl_demo():
    """Performs a demo classification on stl
    """
    logging.info('Loading stl data...')
    stl = visiondata.STL10Dataset(FLAGS.root, 'unlabeled', target_size=32)
    stl_train = visiondata.STL10Dataset(FLAGS.root, 'train', target_size=32)
    stl_test = visiondata.STL10Dataset(FLAGS.root, 'test', target_size=32)

    conv = pipeline.ConvLayer([
            pipeline.PatchExtractor([6, 6], 1), # extracts patches
            pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
            pipeline.LinearEncoder({},
                    trainer = pipeline.ZcaTrainer({'reg': 0.1})),
            pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': False},
                    trainer = pipeline.NormalizedKmeansTrainer(
                         {'k': FLAGS.fromdim, 'max_iter':100})),
            pipeline.SpatialPooler({'grid': (FLAGS.grid, FLAGS.grid), 'method': FLAGS.method}) # average pool
            ], fixed_size = True)
    logging.info('Training the pipeline...')
    conv.train(stl, 400000, exhaustive = True)
    
    logging.info('Extracting features...')
    X = conv.process_dataset(stl, as_2d = False)
    Xtrain = conv.process_dataset(stl_train, as_2d = False)
    Ytrain = stl_train.labels().astype(np.int)
    Xtest = conv.process_dataset(stl_test, as_2d = False)
    Ytest = stl_test.labels().astype(np.int)
    
    X.resize(np.prod(X.shape[:-1]), X.shape[-1])
    Xtrain.resize(np.prod(Xtrain.shape[:-1]), Xtrain.shape[-1])
    Xtest.resize(np.prod(Xtest.shape[:-1]), Xtest.shape[-1])
    # do normalization
    m, std = classifier.feature_meanstd(X, reg=0.01)
    X -= m
    X /= std
    Xtrain -= m
    Xtrain /= std
    Xtest -= m
    Xtest /= std

    covmat = mathutil.mpi_cov(X)
    
    current_dim = FLAGS.fromdim
    if FLAGS.svd == 1:
        eigval, eigvec = np.linalg.eigh(covmat)
    while current_dim >= 100:
        if current_dim < FLAGS.fromdim:
            if FLAGS.svd == 1:
                # directly do dimensionality reduction
                U = eigvec[:, -current_dim:]
                Xtrain_red = np.dot(Xtrain, U)
                Xtest_red = np.dot(Xtest, U)
            else:
                # do subsampling
                temp = code_ap.code_af(X, current_dim, corr = False, tol=current_dim * 0.01)
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

if __name__ == "__main__":
    gflags.FLAGS(sys.argv)
    mpi.root_log_level(logging.INFO)
    stl_demo()
