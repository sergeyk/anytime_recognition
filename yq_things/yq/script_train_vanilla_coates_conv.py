import cPickle as pickle
import logging
from iceberk import mpi, datasets, visiondata, pipeline, classifier
import numpy as np
import os
import sys

mpi.root_log_level(logging.DEBUG)

logging.info('Loading the dataset...')
ilsvrc = visiondata.ILSVRCDataset('/u/vis/x1/common/ILSVRC-2010/train/', 
                                  ['jpeg'], prefetch=False, center_crop = 256) 

conv = pipeline.ConvLayer([
        pipeline.PatchExtractor([6,6], 1), # extracts patches
        pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
        pipeline.LinearEncoder({},
                trainer = pipeline.ZcaTrainer({'reg': 0.1})), # Does whitening
    pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': True},
                trainer = pipeline.OMPTrainer(
                        {'k': 1600, 'max_iter':100})), # does encoding
        pipeline.SpatialPooler({'grid': (2,2), 'method': 'ave'}) # average pool
        ])

logging.info('Training the pipeline...')
conv.train(ilsvrc, 400000)
logging.info('Dumping the pipeline...')
mpi.root_pickle(conv, 'ilsvrc_vanilla_conv.pickle')
