import cPickle as pickle
import coclassification as co
import glob
import h5py
import numpy as np
import os, sys
import gflags

########
# Settings
########

gflags.DEFINE_string("name", "", "The subcategory name")
gflags.DEFINE_bool("baseline", False, "If set, carry out the baseline only")
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

TRAINDIR = '/u/vis/x1/common/ILSVRC-2010/LLC-SIFT/train/'
FEATDIR = "/u/vis/x1/jiayq/ILSVRC/subcategory/" + FLAGS.name + '/'
MODELNAME = "/u/vis/x1/jiayq/ILSVRC/subcategory/" + FLAGS.name \
        + "/ILSVRC-2010-LLC-SIFT-subcategory-model.npz"
TESTFILES = glob.glob(\
        '/u/vis/x1/common/ILSVRC-2010/LLC-SIFT/test/*.mat')
TESTFILES.sort()
TESTLABELS = np.load(\
        '/u/vis/x1/common/ILSVRC-2010/LLC-SIFT/test/labels_ascii_sorted.npy')
# baseline output
BASELINE = np.load('/u/vis/x1/jiayq/ILSVRC/test_predict/test_prob.npy')
CONF = np.load('/u/vis/x1/jiayq/ILSVRC/adagrad_confmats_val.npz')['confmats'][0]
CONF = co.Concept.perform_smoothing(CONF, ('kneserney', 0.75))
# convert to posterior
CONF /= CONF.sum(0)

all_classes = [s[len(TRAINDIR):-4] for s in \
        glob.glob(TRAINDIR + '*.mat')]
all_classes.sort()
sub_classes = [s[len(FEATDIR):-4] for s in glob.glob(FEATDIR + 'n*.mat')]
sub_classes.sort()
# find the slice of the classes
# for the new label value i, its original label value is slice[i]
slice = np.array([all_classes.index(s) for s in sub_classes])
slice_set = set(slice)
TESTMASK = np.array([v in slice_set for v in TESTLABELS])
BASELINE = BASELINE[TESTMASK]
TESTLABELS = TESTLABELS[TESTMASK]
N = float(TESTMASK.sum())

########
# Main script
########
if not FLAGS.baseline:
    model = np.load(MODELNAME)
    w = model['w']
    b = model['b']
    preds = []
    for f in TESTFILES:
        print f
        fid = h5py.File(f, 'r')
        features = np.array(fid['features'])
        pred = np.dot(features, w)
        pred += b
        preds.append(pred.argmax(1))
    preds = np.hstack(preds)
    preds_mapped = slice[preds]
    preds_mapped = preds_mapped[TESTMASK]
    correct = (TESTLABELS == preds_mapped)
    print 'Accuracy', sum(correct) / N

print 'Naive accuracy', sum(BASELINE.argmax(1) == TESTLABELS) / N
print 'Forced choice accuracy', sum(slice[BASELINE[:, slice].argmax(1)] == TESTLABELS) / N 
prob_ys = np.dot(BASELINE, CONF[slice].T)
print 'Adapted algorithm accuracy', sum(slice[prob_ys.argmax(1)] == TESTLABELS) / N
