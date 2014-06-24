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
                         {'k': 100, 'max_iter':100})),
            pipeline.SpatialPooler({'grid': (FLAGS.grid, FLAGS.grid), 'method': FLAGS.method}) # average pool
            ])
    logging.info('Training the pipeline...')
    conv.train(cifar, 400000, exhaustive = True)
    logging.info('Dumping the pipeline...')
    if mpi.is_root():
        with open(os.path.join(FLAGS.output_dir, FLAGS.model_file),'w') as fid:
            pickle.dump(conv, fid)
            fid.close()
    
    logging.info('Extracting features...')
    Xtrain = conv.process_dataset(cifar, as_2d = True)
    Ytrain = cifar.labels().astype(np.int)
    Xtest = conv.process_dataset(cifar_test, as_2d = True)
    Ytest = cifar_test.labels().astype(np.int)

    # normalization
    m, std = classifier.feature_meanstd(Xtrain, reg = 0.01)
    # to match Adam Coates' pipeline
    Xtrain -= m
    Xtrain /= std
    Xtest -= m
    Xtest /= std
    
    """
    covmat = mathutil.mpi_cov(Xtrain)
    eigval, eigvec = np.linalg.eigh(covmat)
    U = eigvec[:,-400:] * np.sqrt(eigval[-400:])
    logging.info("Dump oriol")
    mpi.root_pickle((eigval, eigvec), 'cifar_dump_oriol.pickle')
    Xtrain = np.dot(Xtrain, U)
    Xtest = np.dot(Xtest, U)
    """
    
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.002,
                                     fminargs={'maxfun': 4000})
    if mpi.is_root():
        with open(os.path.join(FLAGS.output_dir, FLAGS.svm_file), 'w') as fid:
            pickle.dump({'m': m, 'std': std, 'w': w, 'b': b}, fid)
    
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
