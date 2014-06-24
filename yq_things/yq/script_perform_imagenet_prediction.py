"""Perform imagenet prediction. This script is written to run on a single machine
"""
from iceberk import mpi
import numpy as np
import os, sys
import gflags
import glob
import h5py

gflags.DEFINE_string("model", "", "The model file")
gflags.DEFINE_string("folder", "", "The input folder that contains the data")
gflags.DEFINE_string("output", "", "The output folder")
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

if FLAGS.folder == "" or FLAGS.model == "":
    sys.exit(1)

model = np.load(FLAGS.model)
w = model['w']
b = model['b']

if not os.path.exists(FLAGS.output):
    mpi.mkdir(FLAGS.output)

files = glob.glob(os.path.join(FLAGS.folder, '*.mat'))
files.sort()

for i in range(mpi.RANK, len(files), mpi.SIZE):
    file = files[i]
    print '%d / %d: %s' % (i, len(files), file)
    fid = h5py.File(file, 'r')
    features = fid['features']
    pred = np.dot(features, w)
    pred += b
    fidout = h5py.File(os.path.join(FLAGS.output, os.path.basename(file)), 'w')
    fidout['pred'] = pred
    fid.close()
    fidout.close()

