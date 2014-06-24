#!/usr/bin/env python
import logging
import numpy as np
import os, sys
import gflags
import glob
import h5py

gflags.DEFINE_string("folder", "/u/vis/x1/jiayq/ILSVRC/ILSVRC-2010-LLC-SIFT-train-predict/", "The input folder that contains the training data")
gflags.DEFINE_string("output", "/u/vis/x1/jiayq/ILSVRC/ILSVRC-2010-LLC-SIFT-train-predict/", "The output folder")
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

files = glob.glob(os.path.join(FLAGS.folder, '*.mat'))
files.sort()
labels = []
preds = []
for i,f in enumerate(files):
    print i,f
    fid = h5py.File(f, 'r')
    pred = np.array(fid['pred'])
    fid.close()
    labels.append(np.ones(pred.shape[0], dtype=np.int) * i)
    preds.append(pred.argmax(1))
np.save(os.path.join(FLAGS.output, 'train_pred.npy'), np.hstack(preds))
np.save(os.path.join(FLAGS.output, 'labels_ascii_sorted.npy'), np.hstack(labels))
