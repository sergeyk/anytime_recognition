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
try:
    conv = pickle.load(open('conv.pickle'))
    regions_pooled = mpi.load_matrix_multi(\
                      '/tscratch/tmp/jiayq/pooled_lda/regions_pooled', N = 10)
except IOError:
    # compute the features
    logging.info("Generating the data...")
    bird = visiondata.CUBDataset('/u/vis/x1/common/CUB_200_2011',
            is_training=True, crop = 1.2, prefetch=True, target_size = [256,256])
    regions = pipeline.PatchExtractor([25,25], 1).sample(bird, 100000)
    regions.resize((regions.shape[0],) + (25,25,3))
    regions_data = datasets.NdarraySet(regions)
    try:
        conv = pickle.load(open('conv.pickle'))
    except IOError:
        logging.info("Training the feature extraction pipeline...")
        conv = pipeline.ConvLayer([
                pipeline.PatchExtractor([5, 5], 1), # extracts patches
                pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
                pipeline.LinearEncoder({},
                        trainer = pipeline.ZcaTrainer({'reg': 0.1})),
                #pipeline.SpatialMeanNormalizer({'channels': 3}),
                pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': False},
                        trainer = pipeline.OMPTrainer(
                                {'k': 3200, 'max_iter':100})),
                pipeline.KernelPooler(\
                        {'kernel': pipeline.KernelPooler.kernel_uniform(15),
                         'method': 'max',
                         'stride': 1})
                ],
                fixed_size = True)
        conv.train(regions_data, 400000)
        mpi.root_pickle(conv, "conv.pickle")
    # so let's get the regions' features after pooling.
    regions_pooled = conv.process_dataset(regions_data)
    mpi.dump_matrix_multi(regions_pooled,
                          '/tscratch/tmp/jiayq/pooled_lda/regions_pooled')

logging.info("Feature shape:" + str(regions_pooled.shape[1:]))
std = mathutil.mpi_std(regions_pooled.reshape(regions_pooled.shape[0], \
        np.prod(regions_pooled.shape[1:])))
# compute the std mean
std.resize(np.prod(regions_pooled.shape[1:-1]), regions_pooled.shape[-1])
std = std.mean(axis=0)
std_order = np.argsort(std)

# now, compute the within-class std
regions_pooled_view = regions_pooled.reshape(regions_pooled.shape[0],
        np.prod(regions_pooled.shape[1:-1]), regions_pooled.shape[-1])
within_std_local = regions_pooled_view.var(axis=1)
print within_std_local.shape
within_std = np.sqrt(mathutil.mpi_mean(within_std_local))
within_std_order = np.argsort(within_std)

std_comparison = within_std / (std + 1e-10)
std_comparison_order = np.argsort(std_comparison)

if mpi.is_root():
    pyplot.figure()
    visualize.show_multiple(conv[-2].dictionary[std_order])
    pyplot.savefig("codes_std_ordered.pdf")
    pyplot.figure()
    visualize.show_multiple(conv[-2].dictionary[within_std_order])
    pyplot.savefig("codes_within_std_ordered.pdf")
    pyplot.figure()
    visualize.show_multiple(conv[-2].dictionary[std_comparison_order])
    pyplot.savefig("codes_std_comparison_ordered.pdf")
    pyplot.figure()
    pyplot.plot(std)
    pyplot.show()
mpi.barrier()

