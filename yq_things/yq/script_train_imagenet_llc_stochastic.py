import cPickle as pickle
from iceberk import mpi, classifier, mathutil
import numpy as np
import logging
import os, sys
import time
import gflags

########
# Settings
########
FEATDIR = "/tscratch/tmp/jiayq/ILSVRC-2010-LLC-SIFT-train/"
VALDIR = "/tscratch/tmp/jiayq/ILSVRC-2010-LLC-SIFT-val/"
DUMPNAME = "/u/vis/x1/jiayq/ILSVRC-2010-LLC-SIFT-model.npz"

gflags.DEFINE_bool("load", False, "If set, load the necessary structures and stop.")
gflags.DEFINE_integer("speedtest", 0, "If set, test the speed of the sampler for the given iterations")
gflags.DEFINE_float("reg", 1e-8, "The reg term")
gflags.DEFINE_integer("minibatch", 10000, "The minibatch size")
gflags.DEFINE_bool("svm", False, "If set, run SVM")
gflags.DEFINE_bool("hier", False, "If set, run hierarchical loss")
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

########
# Main script
########
np.random.seed(int(time.time()) + mpi.RANK * 100)
mpi.root_log_level(level=logging.DEBUG)
logging.info("Loading data...")

base_sampler = mathutil.PrefetchFileSampler(\
        [FEATDIR + '*.mat',
         FEATDIR + 'labels_ascii_sorted.npy',
         None])
if FLAGS.svm:
    sampler = mathutil.PostProcessSampler(\
            base_sampler,
            [lambda X: X,
             lambda Y: classifier.to_one_of_k_coding(Y, fill = -1, K = 1000),
             None])
    loss = classifier.Loss2.loss_hinge
if FLAGS.hier:
    logging.info('Perform hierarchical loss.')
    from birdmix import tax
    graph = tax.get_imagenet_taxonomy(1000)
    leaves = [n for n in graph.nodes() if len(graph.successors(n)) == 0]
    leaves.sort()
    leaf2id = dict((n,i) for i, n in enumerate(leaves))
    infogain = tax.pairwise_info_gain(graph)
    # convert to mat
    igmat = np.zeros((1000,1000))
    for key in infogain:
        igmat[leaf2id[key[0]], leaf2id[key[1]]] = infogain[key]
    np.exp(igmat, igmat)
    print igmat.min()
    igmat -= igmat.min()
    # normalize
    igmat /= igmat.sum(1)[:, np.newaxis]
    sampler = mathutil.PostProcessSampler(\
            base_sampler,
            [lambda X: X,
             lambda Y: np.ascontiguousarray(igmat[Y.astype(np.int)]),
             None])
    loss = classifier.Loss2.loss_multiclass_logistic
    DUMPNAME = "/u/vis/x1/jiayq/ILSVRC-2010-LLC-SIFT-model-hier.npz"
else:
    sampler = mathutil.PostProcessSampler(\
            base_sampler,
            [lambda X: X,
             lambda Y: Y.astype(np.int),
             None])
    loss = classifier.Loss2.loss_multiclass_logistic_yvector

#Xval = mpi.load_matrix_multi(VALDIR + 'Xval')
#Yval = mpi.load_matrix(VALDIR + 'labels_ascii_sorted.npy')
#callback = [lambda wb: classifier.Evaluator.accuracy(Yval, \
#                (np.dot(Xval, wb[0]) + wb[1]).argmax(1))]
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
        args = {'mode': 'adagrad', 'base_lr': 0.05, 'minibatch': FLAGS.minibatch,
                'num_iter': 2000, 'callback': callback, 'eta': 1e-8,
                'dump_every': 25, 'dump_name': DUMPNAME})
w,b = solver.solve(sampler, resume = resume)

mpi.root_savez(DUMPNAME[:-4] + ".final.npz", w = w, b = b)
