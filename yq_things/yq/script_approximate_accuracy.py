from iceberk import mathutil, mpi
import logging
import numpy as np
import os, sys
import gflags
import glob
import h5py

gflags.DEFINE_string("hessian", "", "The hessian file")
gflags.DEFINE_string("folder", "", "The input folder that contains the training data")
gflags.DEFINE_string("pred", "", "The input folder that contains the training data predictions")
gflags.DEFINE_string("scale", "1.", "The scale that is applied to the one-step newton update, separated by commas")
gflags.DEFINE_string("validation", "", "The folder that contains the validation data. If given, we will do binary search to find the optimum scale, and will ignore the scale parameter")
gflags.DEFINE_bool("soft", False, "If set, perform soft prediction, i.e., use logistic regression probability outputs") 
gflags.DEFINE_string("output", "", "The output file")
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

if FLAGS.folder == "" or FLAGS.hessian == "":
    sys.exit(1)

logging.getLogger().setLevel(logging.DEBUG)
logging.debug('Loading data...')
hessian = np.load(FLAGS.hessian)
N = hessian['count']
Hwinv = hessian['hw'] + np.finfo(np.float64).eps
Hbinv = hessian['hb'] + np.finfo(np.float64).eps
# compute the inverse
Hwinv = 1. / Hwinv / float(N)
Hbinv = 1. / Hbinv / float(N)

files = glob.glob(os.path.join(FLAGS.folder, '*.mat'))
preds = glob.glob(os.path.join(FLAGS.pred, '*.mat'))
files.sort()
preds.sort()
if all(mpi.COMM.allgather(len(files) != len(preds))):
    logging.error('%s and %s contain different number of files' % (FLAGS.folder, FLAGS.pred))
    sys.exit(1)

if FLAGS.validation != "":
    logging.info("Performing binary search to match per-class accuracy")
    val_prob = np.load(os.path.join(FLAGS.validation, 'val_prob.npy'))
    val_pred = val_prob.argmax(1)
    val_label = np.load(os.path.join(FLAGS.validation, 'labels_ascii_sorted.npy'))
    use_validation = True
    scales = [float(FLAGS.scale.split(',')[0])]
else:
    use_validation = False
    scales = [float(s) for s in FLAGS.scale.split(',')]

confmat_locals = np.zeros((len(scales), 1000,1000))
if mpi.is_root():
    confmats = np.zeros_like(confmat_locals)
else:
    confmats = None
for i in range(mpi.RANK, len(files), mpi.SIZE):
    # for the i-th file the gt label is i
    file = files[i]
    predfile = preds[i]
    fid = h5py.File(file, 'r')
    features = np.array(fid['features'])
    fid.close()
    fid = h5py.File(predfile, 'r')
    pred = np.array(fid['pred'])
    fid.close()
    # compute new prediction
    diff = mathutil.softmax(pred)
    diff[:,i] -= 1
    features **= 2
    weighted_direction = mathutil.dot(features, Hwinv) + Hbinv
    dpred = diff * weighted_direction
    if use_validation:
        iter = 0
        accu = sum((val_pred == i) & (val_label == i)) / float(sum(val_label == i))
        s_low = 0
        s_high = scales[0]
        while iter < 10:
            s = (s_low + s_high) / 2
            newprob = pred + dpred * s
            mathutil.softmax(newprob, out=newprob)
            newpred = newprob.argmax(1)
            my_accu = sum(newpred==i) / float(pred.shape[0])
            if my_accu > accu:
                s_low = s
            else:
                s_high = s
            iter += 1
        if FLAGS.soft:
            confmat_locals[0, i] = newprob.sum(0)
        else:
            confmat_locals[0,i] = np.bincount(newpred, minlength=1000)
        logging.debug('%d / %d: s %f, a %f, val %f' % (i, len(files), s, my_accu, accu))
    else:
        for sid, s in enumerate(scales):
            # get the new prediction
            newprob = pred + dpred * s
            mathutil.softmax(newprob, out=newprob)
            newpred = newprob.argmax(1)
            if FLAGS.soft:
                confmat_locals[sid, i] = newprob.sum(0)
            else:
                confmat_locals[sid,i] = np.bincount(newpred, minlength=1000)
            logging.debug('%d / %d: %s, scale %f, accu %f' % \
                          (i, len(files), file, s, \
                           confmat_locals[sid,i,i] / float(pred.shape[0])))
# prepare to finish - release some resources first
del features, Hwinv, Hbinv
mpi.barrier()
mpi.COMM.Reduce(confmat_locals, confmats)
if mpi.is_root():
    np.savez(FLAGS.output, confmats=confmats, scales=np.array(scales))
    for sid in range(len(scales)):
        # note that we need to compute per-class accuracy
        accuracy = (np.diag(confmats[sid]) / confmats[sid].sum(1).astype(float)).mean()
        print 'Scale %f, accuracy: %f' % (scales[sid], accuracy)
mpi.barrier()
