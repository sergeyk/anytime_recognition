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
from iceberk.experimental import pinker
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

gflags.DEFINE_integer("patch", 0, "")
gflags.DEFINE_integer("grid", 0, "")
gflags.DEFINE_string("method", "", "")
gflags.DEFINE_string("trainer", "", "")
gflags.DEFINE_integer("fromdim", 0, "")
gflags.DEFINE_integer("todim", 0, "")

FLAGS = gflags.FLAGS

def cifar_demo():
    """Performs a demo classification on cifar
    """

    mpi.mkdir(FLAGS.output_dir)
    logging.info('Loading cifar data...')
    cifar = visiondata.CifarDataset(FLAGS.root, is_training=True)
    cifar_test = visiondata.CifarDataset(FLAGS.root, is_training=False)
    
    if FLAGS.trainer == "pink":
        trainer = pinker.SpatialPinkTrainer({'size': (FLAGS.patch, FLAGS.patch), 'reg': 0.1})
    else:
        trainer = pipeline.ZcaTrainer({'reg': 0.1})

    conv = pipeline.ConvLayer([
            pipeline.PatchExtractor([FLAGS.patch, FLAGS.patch], 1), # extracts patches
            pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
            pipeline.LinearEncoder({},
                    trainer = trainer),
            pipeline.ThresholdEncoder({'alpha': 0.0, 'twoside': False},
                    trainer = pipeline.OMPTrainer(
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
    if False:
        # directly do dimensionality reduction
        eigval, eigvec = np.linalg.eigh(covmat)
        U = eigvec[:, -FLAGS.todim:]
        Xtrain = np.dot(Xtrain, U)
        Xtest = np.dot(Xtest, U)
    else:
        # do subsampling
        import code_ap
        temp = code_ap.code_af(Xtrain, FLAGS.todim)
        sel = temp[0]
        sel = mpi.COMM.bcast(sel)
        Cpred = covmat[sel]
        Csel = Cpred[:,sel]
        W = np.linalg.solve(Csel, Cpred)
        # perform svd
        U, D, _ = np.linalg.svd(W, full_matrices = 0)
        U *= D
        Xtrain = np.dot(Xtrain[:, sel], U)
        Xtest = np.dot(Xtest[:, sel], U)
    Xtrain.resize(Ytrain.shape[0], Xtrain.size / Ytrain.shape[0])
    Xtest.resize(Ytest.shape[0], Xtest.size / Ytest.shape[0])
    
    
    """
    # This part is used to do post-pooling over all features nystrom subsampling
    # normalization
    Xtrain.resize(Xtrain.shape[0], np.prod(Xtrain.shape[1:]))
    Xtest.resize(Xtest.shape[0], np.prod(Xtest.shape[1:]))
    m, std = classifier.feature_meanstd(Xtrain, reg = 0.01)
    # to match Adam Coates' pipeline
    Xtrain -= m
    Xtrain /= std
    Xtest -= m
    Xtest /= std
    
    covmat = mathutil.mpi_cov(Xtrain)
    eigval, eigvec = np.linalg.eigh(covmat)
    U = eigvec[:, -(200*FLAGS.grid*FLAGS.grid):]
    #U = eigvec[:,-400:] * np.sqrt(eigval[-400:])
    Xtrain = np.dot(Xtrain, U)
    Xtest = np.dot(Xtest, U)
    """
    
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.002,
                                     fminargs={'disp': 0, 'maxfun': 1000})
    accu_train = classifier.Evaluator.accuracy(Ytrain, np.dot(Xtrain, w) + b)
    accu_test = classifier.Evaluator.accuracy(Ytest, np.dot(Xtest, w) + b)
    logging.info('Training accuracy: %f' % accu_train)
    logging.info('Testing accuracy: %f' % accu_test)

if __name__ == "__main__":
    gflags.FLAGS(sys.argv)
    if mpi.is_root():
        logging.basicConfig(level=logging.DEBUG)
        if FLAGS.profile_file != "":
            cProfile.run('cifar_demo()', FLAGS.profile_file)
        else:
            cifar_demo()
    else:
        cifar_demo()
