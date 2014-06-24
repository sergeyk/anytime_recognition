import cPickle as pickle
from iceberk import mpi, pipeline, visiondata, mathutil
from jiayq.experiments.feature_selection import pcfs
import logging
import numpy as np
import os

if mpi.is_root():
    logging.basicConfig(level=logging.DEBUG)

stl_folder = '/u/vis/x1/common/STL_10/stl10_matlab'

NUM_REDUCED_DICT = 64
model_file_first = '/u/vis/ttmp/jiayq/stl/conv.pickle'
model_file_second = '/u/vis/ttmp/jiayq/stl/conv_second.pickle'
order_file = '/u/vis/ttmp/jiayq/stl/order.npy'
covmat_file = '/u/vis/ttmp/jiayq/stl/covmat.npy'
mpi.mkdir('/u/vis/ttmp/jiayq/stl/')

logging.info("Loading stl dataset...")
stl = visiondata.STL10Dataset(stl_folder, 'unlabeled')

################################################################################
# Train the first layer 
################################################################################
if os.path.exists(model_file_first):
    logging.info("skipping the first layer training...")
    conv = pickle.load(open(model_file_first,'r'))
else:
    logging.info("Setting up the convolutional layer...")
    conv = pipeline.ConvLayer([
            pipeline.PatchExtractor([5, 5], 1),
            pipeline.MeanvarNormalizer({'reg': 10}),
            pipeline.LinearEncoder({},
                    trainer = pipeline.ZcaTrainer({'reg': 0.01})),
            pipeline.ThresholdEncoder({'alpha':0.0, 'twoside': False},
                    trainer = pipeline.OMPTrainer({'k': 1600})),
            pipeline.KernelPooler({'kernel': pipeline.KernelPooler.kernel_gaussian(10, 5),
                                   'stride': 5, 'method': 'ave'})
            ])
    conv.train(stl, 400000)
    if mpi.is_root():
        fid = open(model_file_first,'w')
        pickle.dump(conv, fid)
        fid.close()
    mpi.barrier()

################################################################################
# Obtains statistics from the first layer
################################################################################
if os.path.exists(order_file):
    logging.info('skipping the feature selection layer...')
    order = np.load(order_file)
else:
    # now, since we cannot possibly store the stl intermediate features, we do 
    # computation online and discard them on the fly
    feat = conv.sample(stl, 400000)
    covmat = mathutil.mpi_cov(feat)
    if mpi.is_root():
        # do greedy feature scoring
        order = pcfs.principal_component_feature_selection(covmat, feat.shape[1])
        np.save(order_file, order)
        np.save(covmat_file, covmat)
        residual = [np.diag(pcfs.conditional_covariance(covmat, order[:i])).sum() \
                    for i in range(1, feat.shape[1], 10)]
        try:
            from matplotlib import pyplot
            pyplot.plot(range(1, feat.shape[1], 10), residual)
            pyplot.show()
        except Exception, e:
            pass
    else:
        order = None
    order = mpi.COMM.bcast(order)
    mpi.barrier()

if os.path.exists(model_file_second):
    logging.info('skipping the second layer model file computation')
    conv2 = pickle.load(open(model_file_second, 'r'))
else:
    logging.info("Setting up the second layer convolutional layer...")
    conv[3].dictionary = np.ascontiguousarray(
            conv[3].dictionary[order[:NUM_REDUCED_DICT]])
    conv2 = pipeline.ConvLayer([
            pipeline.PatchExtractor([4,4], 1),
            pipeline.MeanvarNormalizer({'reg': 0.1}),
            pipeline.LinearEncoder({},
                    trainer = pipeline.ZcaTrainer({'reg': 0.01})),
            pipeline.ThresholdEncoder({'alpha':0.0, 'twoside': False},
                    trainer = pipeline.OMPTrainer({'k': 1600})),
            pipeline.SpatialPooler({'grid': (4,4), 'method': 'max'})
            ], prev = conv)
    conv2.train(stl, 400000)
    if mpi.is_root():
        fid = open(model_file_second,'w')
        pickle.dump(conv2, fid)
        fid.close()
    mpi.barrier()
