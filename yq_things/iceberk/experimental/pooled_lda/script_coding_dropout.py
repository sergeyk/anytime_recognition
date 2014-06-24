'''
Created on Jan 16, 2013

@author: jiayq
'''
import cPickle as pickle
import logging
from matplotlib import pyplot
from iceberk import mpi, visiondata, pipeline, datasets, mathutil, visualize
import numpy as np

mpi.root_log_level(logging.DEBUG)
# compute the features

bird = visiondata.CUBDataset('/u/vis/x1/common/CUB_200_2011',
        is_training=True, crop = 1.2, prefetch=True, target_size = [256,256])
logging.info("Generating the data...")
regions = pipeline.PatchExtractor([5,5], 1).sample(bird, 400000)
normalizer = pipeline.MeanvarNormalizer({'reg': 10})
whitener = pipeline.LinearEncoder({},
        trainer = pipeline.ZcaTrainer({'reg': 0.1}))
encoder = pipeline.ReLUEncoder({'twoside': False},
        trainer = pipeline.OMPTrainer({'k': 1600, 'max_iter':100}))
regions_n = normalizer.process(regions)
whitener.train(regions_n)
regions_w = whitener.process(regions_n)
encoder.train(regions_w)
regions_e = encoder.process(regions_w)
dictionary = encoder.dictionary

ndropout = 100
regions_dropout_m = np.zeros(regions_e.shape)
regions_dropout_std = np.zeros(regions_e.shape)
for i in range(ndropout):
    mask = (np.random.rand(5,5) > 0.5).flatten()
    mask = np.tile(mask[:, np.newaxis], (1,3)).flatten()
    regions_d = regions_w * mask
    regions_dropout = encoder.process(regions_d)
    regions_dropout_m += regions_dropout
    regions_dropout_std += regions_dropout**2
regions_dropout_m /= ndropout
regions_dropout_std /= ndropout
regions_dropout_std -= regions_dropout_m**2
regions_dropout_std = np.sqrt(regions_dropout_std)

