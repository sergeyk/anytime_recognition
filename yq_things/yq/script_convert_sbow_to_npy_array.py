"""This script converts the matlab sbow files downloaded from the ILSVRC 
2010 webpage to the format that our program takes. The labels will be
generated using the string order of the WNIDs - note that this is essentially
different from the label numbers given by the dev toolkit.
"""

import numpy as np
import os
from scipy import io
import glob
import logging

logging.getLogger().setLevel(logging.INFO)

##########
# settings
##########

SRC_DIR = '/u/vis/x1/common/ILSVRC-2010/SBOW/'
DIM = 1000
TARGET_DIR = '/tscratch/tmp/sergeyk/imagenet-sbow/'

##########
# functions
##########

def process_file(filename):
    matdata = io.loadmat(filename)
    image_sbow = matdata['image_sbow']
    num_data = len(image_sbow)
    X = np.zeros((num_data, DIM))
    for i in range(num_data):
        sbow = image_sbow[i]['sbow'][0]['word'][0][0].flatten()
        sbow = np.bincount(sbow)
        if (len(sbow) > 1000):
            raise RuntimeError, 'Found an invalid bincount in ' + filename
        X[i,:len(sbow)] = sbow
    return X

logging.info("Processing validation data")
try:
    os.makedirs(TARGET_DIR + 'val')
except OSError:
    pass
files = glob.glob(SRC_DIR + 'val/*.mat')
files.sort()
X = []
for i, file in enumerate(files):
    X.append(process_file(file))
np.save(TARGET_DIR + 'val/Xval.npy', np.vstack(X))

logging.info("Processing testing data")
try:
    os.makedirs(TARGET_DIR + 'test')
except OSError:
    pass
files = glob.glob(SRC_DIR + 'test/*.mat')
files.sort()
X = []
for i, file in enumerate(files):
    X.append(process_file(file))
np.save(TARGET_DIR + 'test/Xtest.npy', np.vstack(X))

logging.info("Processing training data")
try:
    os.makedirs(TARGET_DIR + 'train')
except OSError:
    pass
files = glob.glob(SRC_DIR + 'train/*.mat')
files.sort()
X = []
Y = []
# There are 1000 classes, and we store them to 10 chunks
for i, file in enumerate(files):
    X.append(process_file(file))
    Y.append(np.ones(X[-1].shape[0], dtype=np.int) * i)
    if (i + 1) % 100 == 0:
        np.save(TARGET_DIR + 'train/Xtrain-%05d-of-%05d.npy' % \
                ((i+1) / 100 - 1, 10), np.vstack(X))
        X = []
np.save(TARGET_DIR + 'train/Ytrain.npy', np.hstack(Y))

