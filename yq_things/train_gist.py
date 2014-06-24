from iceberk import mpi, classifier, mathutil
import numpy as np
import logging
import os
import sys
import time
import gflags
import glob
import json
import itertools
from numpy.core.umath_tests import inner1d

# Settings
TRAINDIR = '/u/vis/x1/jiayq/ILSVRC/gist/train/'
VALDIR = '/u/vis/x1/jiayq/ILSVRC/gist/val/'
TRAIN_LABEL = '/u/vis/x1/jiayq/ILSVRC/train_predict/labels_ascii_sorted.npy'
VAL_LABEL = '/u/vis/x1/jiayq/ILSVRC/val_predict/labels_ascii_sorted.npy'
DUMPNAME = '/u/vis/x1/sergeyk/ILSVRC-2010-GIST-model.npz'

gflags.DEFINE_float("reg", 1e-8, "The reg term")
gflags.DEFINE_integer("minibatch", 10000, "The minibatch size")
gflags.DEFINE_bool("svm", False, "If set, run SVM")
gflags.DEFINE_bool("hier", False, "If set, run hierarchical loss")
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

#
# Main script
#
np.random.seed(int(time.time()) + mpi.RANK * 100)
mpi.root_log_level(level=logging.DEBUG)
logging.info("Loading data...")

base_sampler = mathutil.FileSampler([TRAINDIR + '*.npy', TRAIN_LABEL, None])

if FLAGS.svm:
    sampler = mathutil.PostProcessSampler(
        base_sampler,
        [lambda X: X.astype('float64'),
         lambda Y: classifier.to_one_of_k_coding(Y, fill=-1, K=1000),
         None])
    loss = classifier.Loss2.loss_hinge
if FLAGS.hier:
    logging.info('Perform hierarchical loss.')
    from birdmix import tax
    graph = tax.get_imagenet_taxonomy(1000)
    leaves = [n for n in graph.nodes() if len(graph.successors(n)) == 0]
    leaves.sort()
    leaf2id = dict((n, i) for i, n in enumerate(leaves))
    infogain = tax.pairwise_info_gain(graph)
    # convert to mat
    igmat = np.zeros((1000, 1000))
    for key in infogain:
        igmat[leaf2id[key[0]], leaf2id[key[1]]] = infogain[key]
    np.exp(igmat, igmat)
    print igmat.min()
    igmat -= igmat.min()
    # normalize
    igmat /= igmat.sum(1)[:, np.newaxis]
    sampler = mathutil.PostProcessSampler(
        base_sampler,
        [lambda X: X.astype('float64') / inner1d(X, X)[:, np.newaxis],
         lambda Y: np.ascontiguousarray(
         igmat[Y.astype(np.int)]),
         None])
    loss = classifier.Loss2.loss_multiclass_logistic
    DUMPNAME = "/u/vis/x1/jiayq/ILSVRC-2010-LLC-SIFT-model-hier.npz"
else:
    sampler = mathutil.PostProcessSampler(
        base_sampler,
        [lambda X: X.astype('float64') / np.sqrt(inner1d(X, X)[:, np.newaxis]),
         lambda Y: Y.astype(np.int),
         None])
    loss = classifier.Loss2.loss_multiclass_logistic_yvector


files = glob.glob(VALDIR + '/*.npy')
Xval = np.vstack([np.load(f) for f in sorted(files)]).astype('float64')
Xval /= np.sqrt(inner1d(Xval, Xval)[:, np.newaxis])
Yval = mpi.load_matrix(VAL_LABEL)
callback = [lambda wb: classifier.Evaluator.accuracy(
    Yval, (np.dot(Xval, wb[0]) + wb[1]).argmax(1))]

logging.info("Performing classification")
if os.path.exists(DUMPNAME):
    resume = DUMPNAME
else:
    resume = None

# Do search for reg, base_lr
param_grid = {
    'base_lr': [.1, .01, .001, .0001],
    'reg': [1e-3, 1e-5, 1e-8, 1e-10]
}
param_settings = [dict(zip(('base_lr', 'reg'), x)) for x in list(itertools.product(param_grid['base_lr'], param_grid['reg']))]

accuracies = []
for setting in param_settings:
    print(setting)
    solver = classifier.SolverStochastic(
        setting['reg'], loss, classifier.Reg.reg_l2, args={
            'mode': 'adagrad', 'base_lr': setting['base_lr'],
            'minibatch': FLAGS.minibatch,
            'num_iter': 20, 'callback': callback, 'eta': 1e-8,
            'dump_every': 100, 'dump_name': DUMPNAME})
    w, b = solver.solve(sampler, resume=resume)
    accuracies.append(classifier.Evaluator.accuracy(
        Yval, (np.dot(Xval, w) + b).argmax(1)))
print(accuracies)
max_ind = np.argmax(accuracies)
best_setting = param_settings[max_ind]
with open(DUMPNAME[:-4] + '_params.json', 'w') as f:
    json.dump(best_setting, f)

solver = classifier.SolverStochastic(
    best_setting['reg'], loss, classifier.Reg.reg_l2, args={
        'mode': 'adagrad', 'base_lr': best_setting['base_lr'],
        'minibatch': FLAGS.minibatch,
        'num_iter': 2000, 'callback': callback, 'eta': 1e-8,
        'dump_every': 100, 'dump_name': DUMPNAME})
w, b = solver.solve(sampler, resume=resume)
mpi.root_savez(DUMPNAME[:-4] + ".final.npz", w=w, b=b)
