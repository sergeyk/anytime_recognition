'''
This script uses the iceberk pipeline to perform a cifar classification demo
using parameter settings idential to Adam Coates' AISTATS paper (except for the
number of kmeans centers, which we set to 800 for speed considerations).

You need to specify "--root=/path/to/cifar-data" to run the code. For other
optional flags, run the script with --help or --helpshort.

@author: jiayq
'''

import cPickle as pickle
import logging
from iceberk import mpi, visiondata, pipeline, classifier, datasets, mathutil
from iceberk.experimental import code_ap
import numpy as np

fromdim = 3200
todim = 256

mpi.root_log_level(logging.DEBUG)

data_root = '/u/vis/x1/common/CIFAR/cifar-10-batches-py/'
logging.info('Loading cifar data...')
cifar = visiondata.CifarDataset(data_root, is_training=True)

"""
conv = pipeline.ConvLayer([
        pipeline.PatchExtractor([8,8], 1), # extracts patches
        pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
        pipeline.LinearEncoder({},
                trainer = pipeline.ZcaTrainer({'reg': 0.1})),
        pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': False},
                trainer = pipeline.NormalizedKmeansTrainer(
                     {'k': fromdim, 'max_iter':100})),
                #trainer = pipeline.OMPNTrainer(
                #      {'k': 3200, 'num_active': 10, 'max_iter':100})),
        pipeline.SpatialPooler({'grid': (2,2), 'method': 'max'}) # average pool
        ])
logging.info('Training the pipeline...')
conv.train(cifar, 400000, exhaustive = True)
"""
conv = pickle.load(open('cvpr_exemplar_centroids_conv.pickle'))
_, ap_result = pickle.load(open('cvpr_exemplar_centroids.pickle'))

logging.info('Extracting features...')
Xtrain = conv.process_dataset(cifar, as_2d = False)
# we simply use all the features to compute the covmat
Xtrain.resize(np.prod(Xtrain.shape[:-1]), Xtrain.shape[-1])

m, std = classifier.feature_meanstd(Xtrain, 0.01)
Xtrain -= m
Xtrain /= std
covmat = mathutil.mpi_cov(Xtrain)

# do subsampling
"""
ap_result = code_ap.code_af(Xtrain, todim)
"""
sel = ap_result[0]
sel = mpi.COMM.bcast(sel)
Cpred = covmat[sel]
Csel = Cpred[:,sel]
Crecon = np.dot(Cpred.T, np.dot(np.linalg.pinv(Csel), Cpred))
Crecon = (Crecon + Crecon.T) / 2
eigval = np.linalg.eigvals(covmat)
eigval_recon = np.linalg.eigvals(Crecon)

# random
eigval_random = np.zeros_like(eigval_recon)
for i in range(10):
    sel = np.arange(covmat.shape[0])
    np.random.shuffle(sel)
    sel = sel[:todim]
    Cpred = covmat[sel]
    Csel = Cpred[:,sel]
    Crecon = np.dot(Cpred.T, np.dot(np.linalg.pinv(Csel), Cpred))
    Crecon = (Crecon + Crecon.T) / 2
    eigval_temp = np.linalg.eigvals(Crecon)
    eigval_random += np.sort(eigval_temp)

eigval_random = mpi.COMM.allreduce(eigval_random)
eigval_random /= 10 * mpi.SIZE

"""
mpi.root_pickle(conv, 'cvpr_exemplar_centroids_conv.pickle')
mpi.root_pickle((conv[-2].dictionary, ap_result), 'cvpr_exemplar_centroids.pickle')
"""
mpi.root_pickle((eigval, eigval_recon, eigval_random), 'cvpr_exemplar_centroids_covmat_eigvals.pickle')

# perform sampling
# sample post-pooling guys
Xtrain *= std
Xtrain += m
sampler = mathutil.ReservoirSampler(2000)
for i in range(covmat.shape[0]):
    label = ap_result[1][i]
    centroid_id = ap_result[0][label]
    if centroid_id != i:
        sampler.consider(Xtrain[:, [i, centroid_id]])
mpi.dump_matrix(sampler.get(), 'cvpr_exemplar_centroids_distribution_within_cluster_postpooling.npy')
sampler = mathutil.ReservoirSampler(2000)
for i in range(len(ap_result[0])):
    for j in range(i+1, len(ap_result[0])):
        sampler.consider(Xtrain[:,[ap_result[0][i],ap_result[0][j]]])
mpi.dump_matrix(sampler.get(), 'cvpr_exemplar_centroids_distribution_between_cluster_postpooling.npy')

# clean up something for the large sampler
del Xtrain
del Cpred
del Csel
del Crecon
del sampler

sampler = mathutil.ReservoirSampler(2000)
temp = pipeline.ConvLayer(conv[:-1]).sample(cifar, 200000, True)
for i in range(covmat.shape[0]):
    label = ap_result[1][i]
    centroid_id = ap_result[0][label]
    if centroid_id != i:
        sampler.consider(temp[:, [i, centroid_id]])
mpi.dump_matrix(sampler.get(), 'cvpr_exemplar_centroids_distribution_within_cluster_prepooling.npy')

