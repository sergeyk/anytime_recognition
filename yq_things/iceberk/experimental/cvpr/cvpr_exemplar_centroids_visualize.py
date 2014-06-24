import cPickle as pickle
from iceberk import visualize, mpi
from scipy import misc
import numpy as np

import matplotlib
from matplotlib import pyplot

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

mpi.mkdir('centroids')
mpi.mkdir('distribution')

dictionary, ap_result = pickle.load(open('cvpr_exemplar_centroids.pickle'))
vis = visualize.PatchVisualizer()
im = vis.show_multiple(dictionary, bg_func = np.max)
misc.imsave('centroids/dictionary.png', im)

im = vis.show_multiple(dictionary[ap_result[0]], bg_func = np.max)
misc.imsave('centroids/dictionary_major.png', im)

eigval, eigval_rec, eigval_random = pickle.load(open('cvpr_exemplar_centroids_covmat_eigvals.pickle'))
eigval = np.sort(eigval)[::-1]
eigval_rec = np.sort(eigval_rec)[::-1]
eigval_random = np.sort(eigval_random)[::-1]
fig = pyplot.figure()
pyplot.plot(np.log(eigval[:600]), 'g-', lw=2)
pyplot.plot(np.log(eigval[:256]), 'b-.', lw=2)
pyplot.plot(np.log(eigval_rec[:256]), 'r--', lw=2)
#pyplot.plot(np.log(eigval_random[:256]), 'm:', lw=2)
#pyplot.legend(('eigval', 'eigval pca', 'eigval ap', 'eigval random'), loc='upper right')
pyplot.legend(('eigval', 'eigval pca', 'eigval ap'), loc='upper right')
pyplot.title('Eigenvalues and Approximation')
fig.set_size_inches((6.,4.))
fig.savefig('centroids/eigvals.pdf')

for i, id in enumerate(ap_result[0]):
    # we will visualize each centroid and its 10 most related guys
    print i
    centroid = dictionary[id]
    neighbors = np.flatnonzero(ap_result[1] == i)
    idx = np.argsort(-ap_result[2][id, neighbors])
    # select the 10 closest
    neighbors = neighbors[idx][:11]
    if len(neighbors) > 0:
        im = vis.show_single(centroid)
        misc.imsave('centroids/%d.png' % (i), im)
        im = vis.show_multiple(dictionary[neighbors], 1, np.max)
        misc.imsave('centroids/%d-neighbors.png' % (i), im)

within_cluster_samples = mpi.load_matrix(\
        'cvpr_exemplar_centroids_distribution_within_cluster_postpooling.npy')
between_centroids_samples = mpi.load_matrix(\
        'cvpr_exemplar_centroids_distribution_between_cluster_postpooling.npy')
within_cluster_samples_prepooling = mpi.load_matrix(\
        'cvpr_exemplar_centroids_distribution_within_cluster_prepooling.npy')
corr_within_cluster = np.corrcoef(within_cluster_samples.T)
corr_between_centroids = np.corrcoef(between_centroids_samples.T)
corr_within_cluster_prepooling = np.corrcoef(within_cluster_samples_prepooling.T)
# subsample for visualization
within_cluster_samples = within_cluster_samples[\
        np.random.randint(within_cluster_samples.shape[0], size=2000)]
between_centroids_samples = between_centroids_samples[\
        np.random.randint(between_centroids_samples.shape[0], size=2000)]
within_cluster_samples_prepooling = within_cluster_samples_prepooling[\
        np.random.randint(within_cluster_samples_prepooling.shape[0], size=2000)]


fig = pyplot.figure()
ax = fig.add_subplot(111)
ax.plot(within_cluster_samples[:,0], within_cluster_samples[:,1], 'b.', ms=10, mew=0)
ax.set_aspect('equal')
ax.axis([0,6,0,6])
ax.set_title('$\\rho$ = %.3f' % corr_within_cluster[0,1])
fig.savefig('distribution/within_cluster_postpooling.pdf')

fig = pyplot.figure()
ax = fig.add_subplot(111)
ax.plot(between_centroids_samples[:,0], between_centroids_samples[:,1], 'b.', ms = 10, mew=0)
ax.set_aspect('equal')
ax.set_title('$\\rho$ = %.3f' % corr_between_centroids[0,1])
ax.axis([0,6,0,6])
fig.savefig('distribution/between_centroids_postpooling.pdf')

fig = pyplot.figure()
ax = fig.add_subplot(111)
ax.plot(within_cluster_samples_prepooling[:,0], within_cluster_samples_prepooling[:,1], 'b.', ms = 10, mew=0)
ax.set_aspect('equal')
ax.set_title('$\\rho$ = %.3f' % corr_within_cluster_prepooling[0,1])
ax.axis([0,4,0,4])
fig.savefig('distribution/within_cluster_prepooling.pdf')
