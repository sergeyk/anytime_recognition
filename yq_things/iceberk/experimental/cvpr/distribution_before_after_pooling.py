import cPickle as pickle
from matplotlib import pyplot
from iceberk import visualize, mpi
from scipy import misc
import numpy as np

import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

mpi.mkdir('distribution')

pyplot.ion()

dictionary, before_pooling, after_pooling = pickle.load(open('distribution_before_after_pooling.pickle'))
after_pooling -= after_pooling.min(0)

corr_before = np.corrcoef(before_pooling.T)
corr_after = np.corrcoef(after_pooling.T)

# do random sampling for visualization
before_pooling = before_pooling[np.random.randint(before_pooling.shape[0], size=1000)]
after_pooling = after_pooling[np.random.randint(after_pooling.shape[0], size=1000)]

vis = visualize.PatchVisualizer()
im = vis.show_multiple(dictionary)
misc.imsave('distribution/1.png', im[:8,:8])
misc.imsave('distribution/2.png', im[:8,-8:])
misc.imsave('distribution/3.png', im[-8:,:8])
misc.imsave('distribution/4.png', im[-8:,-8:])

fig = pyplot.figure()
ax = fig.add_subplot(121)
ax.plot(before_pooling[:,0], before_pooling[:,1], 'b.', mew=0)
ax.set_title('Before pooling, $\\rho$ = %.3f' % corr_before[0,1])
ax.set_aspect('equal')
ax.axis([0,4,0,4])
ax = fig.add_subplot(122)
ax.plot(after_pooling[:,0], after_pooling[:,1], 'b.', mew=0)
ax.set_title('After pooling, $\\rho$ = %.3f' % corr_after[0,1])
ax.set_aspect('equal')
ax.axis([0,6,0,6])
fig.savefig('distribution/12_distribution.pdf')

fig = pyplot.figure()
ax = fig.add_subplot(121)
ax.plot(before_pooling[:,2], before_pooling[:,3], 'b.', mew=0)
ax.set_title('Before pooling, $\\rho$ = %.3f' % corr_before[2,3])
ax.set_aspect('equal')
ax.axis([0,4,0,4])
ax = fig.add_subplot(122)
ax.plot(after_pooling[:,2], after_pooling[:,3], 'b.', mew=0)
ax.set_title('After pooling, $\\rho$ = %.3f' % corr_after[2,3])
ax.set_aspect('equal')
ax.axis([0,6,0,6])
fig.savefig('distribution/34_distribution.pdf')
