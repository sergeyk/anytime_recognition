from iceberk import visiondata, pipeline, mpi, datasets, classifier
import cPickle as pickle
import logging
import numpy as np

if mpi.is_root():
    logging.getLogger().setLevel(logging.DEBUG)

# Parameters of the script
ROOT = '/u/vis/farrell/datasets/CUB_200_2011/'
CONVOLUTION_OUTPUT = '/u/vis/ttmp/jiayq/birds/conv/'
CONVOLUTION_FILE = '/u/vis/ttmp/jiayq/birds/conv/convolution.pickle'

CROP = 1.5
MIRRORED = True
SUBSET = None
TARGET_SIZE = [128,128]

CONV = pipeline.ConvLayer(
    [pipeline.PatchExtractor([5,5], 1),
     pipeline.MeanvarNormalizer({'reg': 10}),
     pipeline.LinearEncoder({},
                trainer = pipeline.ZcaTrainer({'reg': 0.1})),
     pipeline.ReLUEncoder({'twoside': False},
                trainer = pipeline.NormalizedKmeansTrainer({'k': 200, 'max_iter': 100})),
     #pipeline.PyramidPooler({'level': 2, 'method': 'ave'})
     pipeline.FixedSizePooler({'size':'5', 'method': 'max'})
    ])
CONV2 = pipeline.ConvLayer(
    [pipeline.PatchExtractor([4,4], 1),
     pipeline.MeanvarNormalizer({'reg': 0.01}),
     pipeline.LinearEncoder({},
                trainer = pipeline.ZcaTrainer({'reg': 0.01})),
     pipeline.ReLUEncoder({'twoside': False},
                trainer = pipeline.NormalizedKmeansTrainer({'k': 1600, 'max_iter': 100})),
     pipeline.PyramidPooler({'level': 3, 'method': 'max'})
    ], prev = CONV)

CONV_SPM_GAMMA = 0.01

logging.debug('Loading data...')
train_data = visiondata.CUBDataset(ROOT, True, crop = CROP, subset = SUBSET,
                                  target_size = TARGET_SIZE, prefetch = True)
test_data = visiondata.CUBDataset(ROOT, False, crop = CROP, subset = SUBSET,
                                 target_size = TARGET_SIZE, prefetch = True)
mpi.mkdir(CONVOLUTION_OUTPUT)
if MIRRORED:
    train_data = datasets.MirrorSet(train_data)
    # note that we do not mirror test data.
logging.debug('Training convolutional NN...')
CONV.train(train_data, 400000, exhaustive = True)
CONV2.train(train_data, 400000, exhaustive = True)

mpi.root_pickle(CONV2, CONVOLUTION_FILE)
Xtrain = CONV2.process_dataset(train_data)
Ytrain = train_data.labels().astype(np.int)
Xtest = CONV2.process_dataset(test_data)
Ytest = test_data.labels().astype(np.int)

# normalization
m, std = classifier.feature_meanstd(Xtrain, reg = 0.01)
# to match Adam Coates' pipeline
Xtrain -= m
Xtrain /= std
Xtest -= m
Xtest /= std

w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.005,
                                 fminargs={'maxfun': 500})

accu_train = classifier.Evaluator.accuracy(Ytrain, np.dot(Xtrain, w) + b)
accu_test = classifier.Evaluator.accuracy(Ytest, np.dot(Xtest, w) + b)
logging.info('Training accuracy: %f' % accu_train)
logging.info('Testing accuracy: %f' % accu_test)
