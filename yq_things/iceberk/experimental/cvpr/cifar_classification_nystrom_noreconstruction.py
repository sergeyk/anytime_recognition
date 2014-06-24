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

gflags.DEFINE_string("root", "",
                     "The root to the cifar dataset (python format)")
gflags.RegisterValidator('root', lambda x: x != "",
                         message='--root must be provided.')
gflags.DEFINE_string("output_dir", ".",
                     "The output directory that stores dumped feature and"
                     "classifier files.")
gflags.DEFINE_string("model_file", "conv.pickle",
                     "The filename to output the CNN model.")
gflags.DEFINE_string("feature_file", "cnn_features",
                     "The filename to output the features.")
gflags.DEFINE_string("svm_file", "svm.pickle",
                     "The filename to output the trained svm parameters.")
gflags.DEFINE_string("profile_file", "",
                     "If set, do cProfile analysis, and output the result"
                     "to the filename given by the flag.")

gflags.DEFINE_integer("grid", 0, "")
gflags.DEFINE_string("method", "", "")
gflags.DEFINE_integer("fromdim", 0, "")
gflags.DEFINE_integer("svd", 0, "")

FLAGS = gflags.FLAGS

def cifar_demo():
    """Performs a demo classification on cifar
    """

    mpi.mkdir(FLAGS.output_dir)
    logging.info('Loading cifar data...')
    cifar = visiondata.CifarDataset(FLAGS.root, is_training=True)
    cifar_test = visiondata.CifarDataset(FLAGS.root, is_training=False)

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
    logging.info('Training the pipeline...')
    conv.train(cifar, 400000, exhaustive = True)
    
    logging.info('Extracting features...')
    Xtrain = conv.process_dataset(cifar, as_2d = False)
    Ytrain = cifar.labels().astype(np.int)
    Xtest = conv.process_dataset(cifar_test, as_2d = False)
    Ytest = cifar_test.labels().astype(np.int)
    
    # before we do feature computation, try to do dimensionality reduction
    Xtrain.resize(np.prod(Xtrain.shape[:-1]), Xtrain.shape[-1])
    Xtest.resize(np.prod(Xtest.shape[:-1]), Xtest.shape[-1])
    
    m, std = classifier.feature_meanstd(Xtrain, 0.01)
    Xtrain -= m
    Xtrain /= std
    Xtest -= m
    Xtest /= std
    
    covmat = mathutil.mpi_cov(Xtrain)
    
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
                temp = code_ap.code_af(Xtrain, current_dim)
                logging.info("selected %d dims" % len(temp[0]))
                sel = temp[0]
                Xtrain_red = np.ascontiguousarray(Xtrain[:, sel])
                Xtest_red = np.ascontiguousarray(Xtest[:, sel])
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
    mpi.root_log_level(logging.DEBUG)
    cifar_demo()
