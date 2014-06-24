import logging
from iceberk import visiondata, pipeline, mpi, classifier
import numpy as np


if mpi.is_root():
    logging.getLogger().setLevel(logging.INFO)

logging.info('Loading cifar data...')
cifar = visiondata.CifarDataset('/u/vis/x1/common/CIFAR/cifar-10-batches-py', \
                                is_training=True)
cifar_test = visiondata.CifarDataset(
        '/u/vis/x1/common/CIFAR/cifar-10-batches-py', \
        is_training=False)

code_sizes = [50,100,200,400,800,1600]
pool_sizes = [1,2,3,4]

accuracy_record = np.zeros((len(code_sizes), len(pool_sizes)))

for cid, code_size in enumerate(code_sizes):
    conv = pipeline.ConvLayer([
        pipeline.PatchExtractor([6,6], 1), # extracts patches
        pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
        pipeline.LinearEncoder({},
                trainer = pipeline.ZcaTrainer({'reg': 0.1})), # Does whitening
        pipeline.ThresholdEncoder({'alpha': 0.0, 'twoside': False},
                trainer = pipeline.NormalizedKmeansTrainer(
                        {'k': code_size, 'max_iter':100})), # does encoding
        pipeline.SpatialPooler({'grid': (2,2), 'method': 'rms'})
    ])
    
    logging.debug('Training the pipeline...')
    conv.train(cifar, 400000, exhaustive=True)
    
    for pid, pool_size in enumerate(pool_sizes):
        conv[-1] = pipeline.SpatialPooler({'grid': (pool_size, pool_size),
                                           'method': 'rms'})
        logging.debug('Extracting features...')
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
        logging.debug('Training accuracy: %f' % accu_train)
        logging.info('code %d, pool %d, testing accuracy: %f' % (code_size, pool_size, accu_test))
        accuracy_record[cid, pid] = accu_test

mpi.root_pickle((code_sizes, pool_sizes, accuracy_record), 'cifar_code_pool_size_comparison.pickle')
