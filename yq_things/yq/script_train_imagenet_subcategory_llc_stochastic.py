import cPickle as pickle
import glob
from iceberk import mpi, classifier, mathutil
import numpy as np
import logging
import os, sys
import time
import gflags

########
# Settings
########

gflags.DEFINE_bool("load", False, "If set, load the necessary structures and stop.")
gflags.DEFINE_integer("speedtest", 0, "If set, test the speed of the sampler for the given iterations")
gflags.DEFINE_float("reg", 1e-8, "The reg term")
gflags.DEFINE_float("base_lr", 0.02, "The base learning rate")
gflags.DEFINE_integer("minibatch", 10000, "The minibatch size")
gflags.DEFINE_string("name", "", "The subcategory name")
gflags.DEFINE_bool("preload", False, "If set, preload all the training data (memory heavy)")
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

FEATDIR = "/tscratch/tmp/jiayq/ILSVRC-subcategory/" + FLAGS.name + '/'
DUMPNAME = "/u/vis/x1/jiayq/ILSVRC/subcategory/" + FLAGS.name \
        + "/ILSVRC-2010-LLC-SIFT-subcategory-model.npz"

########
# Main script
########
np.random.seed(int(time.time()) + mpi.RANK * 100)
mpi.root_log_level(level=logging.DEBUG)
logging.info("Loading data...")

if FLAGS.preload:
    files = glob.glob(FEATDIR + 'n*.mat')
    files.sort()
    if mpi.is_root():
        print 'files', files
    Xtrain = mpi.load_matrix_multi(files, name='features')
    Ytrain = mpi.load_matrix(FEATDIR + 'labels_ascii_sorted.npy').astype(np.int)
    sampler = mathutil.NdarraySampler([Xtrain, Ytrain, None], copy=False)
else:
    base_sampler = mathutil.PrefetchFileSampler(\
            [FEATDIR + '*.mat',
             FEATDIR + 'labels_ascii_sorted.npy',
             None])
    sampler = mathutil.PostProcessSampler(\
            base_sampler,
            [lambda X: X,
            lambda Y: Y.astype(np.int),
            None])

loss = classifier.Loss2.loss_multiclass_logistic_yvector

callback = None

if FLAGS.load:
    logging.info("Loading done.")
    sys.exit(0)

if FLAGS.speedtest > 0:
    logging.info("Testing speed")
    logging.info("minibatch size: %d" % FLAGS.minibatch)
    from iceberk.util import Timer
    timer = Timer()
    for i in range(FLAGS.speedtest):
        batch = sampler.sample(FLAGS.minibatch)
        logging.info("Local size: %d" % batch[0].shape[0])
        total_size = mpi.COMM.allreduce(batch[0].shape[0])
        logging.info("Total size: %d" % total_size)
        logging.info("Sampling took %s secs" % timer.lap())
    sys.exit(0)

logging.info("Performing classification")
if os.path.exists(DUMPNAME):
    resume = DUMPNAME
else:
    resume = None
# adagrad
solver = classifier.SolverStochastic(FLAGS.reg,
        loss,
        classifier.Reg.reg_l2,
        args = {'mode': 'adagrad', 'base_lr': FLAGS.base_lr, 'minibatch': FLAGS.minibatch,
                'num_iter': 1000, 'callback': callback, 'eta': 1e-8,
                'dump_every': 25, 'dump_name': DUMPNAME})
w,b = solver.solve(sampler, resume = resume)

mpi.root_savez(DUMPNAME[:-4] + ".final.npz", w = w, b = b)
