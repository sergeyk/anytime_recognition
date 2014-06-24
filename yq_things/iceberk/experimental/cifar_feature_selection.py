import cPickle as pickle
import logging
from iceberk import visiondata, pipeline, mpi, classifier, mathutil
import numpy as np
import os

from jiayq.experiments.feature_selection import pcfs

if mpi.is_root():
    logging.getLogger().setLevel(logging.DEBUG)

logging.info('Loading cifar data...')
cifar = visiondata.CifarDataset('/u/vis/x1/common/CIFAR/cifar-10-batches-py', \
                                is_training=True)
cifar_test = visiondata.CifarDataset(
        '/u/vis/x1/common/CIFAR/cifar-10-batches-py', \
        is_training=False)

try:
    conv = pickle.load(open('cifar_conv.pickle'))
    logging.info('Skipping first layer training')
except Exception, e:
    conv = pipeline.ConvLayer([
        pipeline.PatchExtractor([6,6], 1), # extracts patches
        pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
        pipeline.LinearEncoder({},
                trainer = pipeline.ZcaTrainer({'reg': 0.1})), # Does whitening
        pipeline.ThresholdEncoder({'alpha': 0.0, 'twoside': False},
                trainer = pipeline.NormalizedKmeansTrainer(
                        {'k': 1600, 'max_iter':100})), # does encoding
        pipeline.SpatialPooler({'grid': (2,2), 'method': 'ave'})
    ])
    
    logging.info('Training the pipeline...')
    conv.train(cifar, 400000, exhaustive=True)
    mpi.root_pickle(conv, 'cifar_conv.pickle')

# do pruning
try:
    selected_idx = pickle.load(open('cifar_selected_idx.pickle'))
    logging.info('Skipping first layer pruning')
except Exception, e:
    features = conv.sample(cifar, 200000, True)
    mpi.dump_matrix_multi(features, '/u/vis/ttmp/jiayq/cifar/cifar_feature_pooled_sample')
    m, std = mathutil.mpi_meanstd(features)
    features -= m
    features /= std
    covmat = mathutil.mpi_cov(features, reg = 0.01)
    if mpi.is_root():
        selected_idx = pcfs.max_variance_feature_selection(covmat, 800)
    else:
        selected_idx = None
    selected_idx = mpi.COMM.bcast(selected_idx)
    mpi.root_pickle((m, std, covmat), 'cifar_squared_correlation.pickle')
    mpi.root_pickle(selected_idx, 'cifar_selected_idx.pickle')
    
dictionary_all = conv[-2].dictionary

for i in [25,50,100,200,400,800,1600]:
    logging.info('Training with dictionary size %d' % i)
    #conv[-2].dictionary = np.ascontiguousarray(dictionary_all[selected_idx[:i]])
    conv[-2].dictionary = np.ascontiguousarray(dictionary_all[:i])

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
    
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.005,
                                     fminargs={'disp': 0, 'maxfun': 1000})
    
    accu_train = classifier.Evaluator.accuracy(Ytrain, np.dot(Xtrain, w) + b)
    accu_test = classifier.Evaluator.accuracy(Ytest, np.dot(Xtest, w) + b)
    logging.info('Training accuracy: %f' % accu_train)
    logging.info('Testing accuracy: %f' % accu_test)
    mpi.root_pickle((m, std, w, b), 'cifar_classifier.pickle')
