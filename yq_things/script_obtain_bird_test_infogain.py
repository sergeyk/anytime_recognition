"""
The main script to test the accuracy on birds
"""

import cPickle as pickle
from iceberk import mpi, visiondata, datasets, pipeline, classifier
import numpy as np
import logging
import os,sys

# local import
import birdtax

########
# Settings
########

FEATDIR = "/u/vis/ttmp/jiayq/birds/"

########
# Main script
########
if mpi.SIZE > 1:
    raise RuntimeError, "This script runs on single machine only."

Xtest = mpi.load_matrix_multi(os.path.join(FEATDIR,'Xtest'))
Ytest = mpi.load_matrix_multi(os.path.join(FEATDIR,'Ytest')).astype(np.int)
infogain = birdtax.bird_info_gain()

info_mean = infogain.mean(axis=1)
info_max = infogain.max(axis=1)

randguess = sum([info_mean[y] for y in Ytest]) / float(len(Ytest))
bestguess = sum([info_max[y] for y in Ytest]) / float(len(Ytest))

print 'Random guess baseline:', randguess
print 'Best guess baseline:', bestguess

for filename in sys.argv[1:]:
    data = pickle.load(open(filename))
    w, b = data[:2]
    accu_test = data[-1]
    pred = np.argmax(np.dot(Xtest, w) + b, axis=1)
    gain = sum([infogain[y,p] for y,p in zip(Ytest, pred)]) / float(len(Ytest))
    print filename, accu_test, gain
