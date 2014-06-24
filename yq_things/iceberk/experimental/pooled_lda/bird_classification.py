'''
This script uses the iceberk pipeline to perform a bird classification demo
using parameter settings idential to Adam Coates' AISTATS paper (except for the
number of kmeans centers, which we set to 800 for speed considerations).

You need to specify "--root=/path/to/bird-data" to run the code. For other
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

gflags.DEFINE_string("root",
                     "/u/vis/x1/common/CUB_200_2011/",
                     "The root to the bird dataset (python format)")
gflags.RegisterValidator('root', lambda x: x != "",
                         message='--root must be provided.')
gflags.DEFINE_string("version", "2011", "")
gflags.DEFINE_bool("mirrored", True, "")
gflags.DEFINE_float("crop", 1.2, "")
gflags.DEFINE_integer("patch", 5, "")
gflags.DEFINE_integer("k", 1600, "") 
FLAGS = gflags.FLAGS

TARGET_SIZE = [128,128]

def bird_demo():
    logging.info('Loading data...')
    bird = visiondata.CUBDataset(FLAGS.root, is_training=True, crop=FLAGS.crop, 
                                 version=FLAGS.version, prefetch=True,
                                 target_size = TARGET_SIZE)
    bird_test = visiondata.CUBDataset(FLAGS.root, is_training=False, crop=FLAGS.crop, 
                                 version=FLAGS.version, prefetch=True,
                                 target_size = TARGET_SIZE)
    if FLAGS.mirrored:
        bird = datasets.MirrorSet(bird)
    conv = pipeline.ConvLayer([
            pipeline.PatchExtractor([FLAGS.patch, FLAGS.patch], 1), # extracts patches
            pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
            pipeline.LinearEncoder({},
                    trainer = pipeline.ZcaTrainer({'reg': 0.1})),
            pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': True},
                    trainer = pipeline.OMPTrainer(
                         {'k': FLAGS.k, 'max_iter':100})),
            pipeline.SpatialPooler({'grid': 4, 'method': 'max'})
            ],
            fixed_size = True)
    logging.info('Training the pipeline...')
    conv.train(bird, 400000, exhaustive = True)
    
    logging.info('Extracting features...')
    Xtrain = conv.process_dataset(bird, as_2d = True)
    Ytrain = bird.labels().astype(np.int)
    Xtest = conv.process_dataset(bird_test, as_2d = True)
    Ytest = bird_test.labels().astype(np.int)

    # normalization
    m, std = classifier.feature_meanstd(Xtrain, reg = 0.01)
    # to match Adam Coates' pipeline
    Xtrain -= m
    Xtrain /= std
    Xtest -= m
    Xtest /= std
    
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.005,
                                     fminargs={'maxfun': 1000})
    accu_train = classifier.Evaluator.accuracy(Ytrain, np.dot(Xtrain, w) + b)
    accu_test = classifier.Evaluator.accuracy(Ytest, np.dot(Xtest, w) + b)
    logging.info('Training accuracy: %f' % accu_train)
    logging.info('Testing accuracy: %f' % accu_test)
    mpi.root_pickle((m, std, w, b, conv[-2].dictionary), 'debug_features.pickle')

if __name__ == "__main__":
    gflags.FLAGS(sys.argv)
    mpi.root_log_level(logging.DEBUG)
    bird_demo()
