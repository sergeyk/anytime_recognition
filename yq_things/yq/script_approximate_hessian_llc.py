from iceberk import mathutil, mpi
import numpy as np
import os, sys
import gflags
import glob
import h5py

gflags.DEFINE_string("model", "", "The model file")
gflags.DEFINE_string("folder", "", "The input folder that contains the data")
gflags.DEFINE_string("output", "", "The output file")
gflags.DEFINE_float("reg", 1e-8, "The regularization term")
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

if FLAGS.folder == "" or FLAGS.model == "":
    sys.exit(1)

model = np.load(FLAGS.model)
w = model['w']
b = model['b']

hw = np.zeros_like(w)
hb = np.zeros_like(b)

files = glob.glob(os.path.join(FLAGS.folder, '*.mat'))
files.sort()

count = 0
for i in range(mpi.RANK, len(files), mpi.SIZE):
    file = files[i]
    print '%d / %d: %s' % (i, len(files), file)
    fid = h5py.File(file, 'r')
    features = np.array(fid['features'])
    count += features.shape[0]
    pred = np.dot(features, w)
    pred += b
    prob = mathutil.softmax(pred)
    dpdp = prob * (1 - prob)
    # the gradient for b is simple
    hb += np.sum(dpdp, 0)
    features **= 2
    hw += mathutil.dot(features.T, dpdp)
    fid.close()
del features
count = mpi.COMM.allreduce(count)
# we no longer need w and b, so we use them as mpi buffer
mpi.COMM.Allreduce(hw, w)
mpi.COMM.Allreduce(hb, b)
if mpi.is_root():
    hw /= count
    hb /= count
    # add regularization term
    hw += 2 * FLAGS.reg
    np.savez(FLAGS.output, count=count, hw=w, hb=b)
