from iceberk import mpi
import logging
import numpy as np
import time

mpi.root_log_level(logging.INFO)

# just a large matrix
a = np.random.rand(10000,10000)
a_local = np.random.rand(10000,10000)
rank = mpi.RANK

logging.info('Testing mpi size %d' % mpi.SIZE)

mpi.barrier()
start = time.time()
a = a_local + a_local
logging.info('One single addition speed: %f s' % (time.time() - start))

mpi.barrier()
start = time.time()
if (rank == 1):
    mpi.COMM.Send(a_local, dest=0)
if (rank == 0):
    mpi.COMM.Recv(a_local, source=1)
logging.info('One single send/recv speed: %f s' % (time.time() - start))

mpi.barrier()
start = time.time()
mpi.COMM.Reduce(a_local, a)
logging.info('Reduce speed: %f s' % (time.time() - start))

mpi.barrier()
start = time.time()
mpi.COMM.Allreduce(a_local, a)
logging.info('Allreduce speed: %f s' % (time.time() - start))
