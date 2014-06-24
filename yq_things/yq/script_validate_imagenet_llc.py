import cPickle as pickle
from iceberk import mpi, classifier, mathutil
import numpy as np
import logging
import os, sys

VALDIR = "/tscratch/tmp/jiayq/ILSVRC-2010-LLC-SIFT-val/"
DUMPNAME = "/u/vis/x1/jiayq/ILSVRC-2010-LLC-SIFT-model.npz"

mpi.root_log_level(logging.DEBUG)

Xval = mpi.load_matrix_multi(VALDIR + 'Xval')
Yval = mpi.load_matrix(VALDIR + 'labels_ascii_sorted.npy')
npzfile = np.load(DUMPNAME)
pred = np.dot(Xval, npzfile['w']) + npzfile['b']
accu = classifier.Evaluator.accuracy(Yval, 
        pred.argmax(1))
accu5 = classifier.Evaluator.top_k_accuracy(Yval, pred, 5)
logging.debug('accu: %f, %f', accu, accu5)

# perform training accuracy 
