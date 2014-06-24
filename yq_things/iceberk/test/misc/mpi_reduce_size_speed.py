from iceberk import mpi
import logging
import numpy as np
import time

mpi.root_log_level(logging.INFO)

# just a large matrix
a = np.random.rand(1000,12800)
a_local = np.random.rand(1000,12800)
rank = mpi.RANK

logging.info('Testing mpi size %d' % mpi.SIZE)

mpi.barrier()
start = time.time()
mpi.COMM.Allreduce(a_local, a)
logging.info('Allreduce big speed: %f s' % (time.time() - start))

mpi.barrier()
start = time.time()
for i in xrange(a.shape[0]):
    mpi.COMM.Allreduce(a_local[i], a[i])
logging.info('Allreduce small speed: %f s' % (time.time() - start))
