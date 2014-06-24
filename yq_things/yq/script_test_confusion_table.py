import cPickle as pickle
from iceberk import classifier, mathutil
import coclassification as co
import glob
import h5py
import numpy as np
import logging
import os, sys

ROOT = '/u/vis/x1/jiayq/ILSVRC/'
eps = np.finfo(np.float64).eps

# first, get all the labels and predictions
train_label = np.load(ROOT + 'train_predict/labels_ascii_sorted.npy')
val_label   = np.load(ROOT + 'val_predict/labels_ascii_sorted.npy')
test_label  = np.load(ROOT + 'test_predict/labels_ascii_sorted.npy')
train_pred  = np.load(ROOT + 'train_predict/train_pred.npy')
val_pred    = np.load(ROOT + 'val_predict/val_pred.npy')
test_pred   = np.load(ROOT + 'test_predict/test_pred.npy')
train_conf  = classifier.Evaluator.confusion_table(train_label, train_pred)
val_conf    = classifier.Evaluator.confusion_table(val_label, val_pred)
test_conf   = classifier.Evaluator.confusion_table(test_label, test_pred)
test_num = test_conf.sum()
laplace_interval = np.logspace(-3,1,17)
#laplace_interval = np.arange(20)

# simple routine to get the log likelihood
normalize = lambda confmat: \
        confmat.astype(np.float64) / confmat.sum(1)[:, np.newaxis]
get_acc = lambda confmat: \
        (np.diag(confmat) / confmat.sum(1).astype(float)).mean()
get_perplexity = lambda confmat: \
        0.5 ** (np.dot(test_conf.flat, \
                       np.log2(normalize(confmat) + eps).flat) \
                / test_num)

# test all methods
# 1.a training confmat, no smoothing
print '1a', get_perplexity(train_conf)
# 1.b training confmat, laplace smoothing
res_1b = np.array([get_perplexity(train_conf + i) for i in laplace_interval])
print '1b', res_1b.min(), 'smoothing =', laplace_interval[res_1b.argmin()]
# 1.c training confmat, Kneser-ney smoothing
res_1c = get_perplexity(co.Concept.perform_smoothing(train_conf, ('kneserney', 0.75, 1)))
print '1c', res_1c

# 2.a validation confmat, no smoothing
print '2a', get_perplexity(val_conf)
# 2.b validation confmat, laplace smoothing
res_2b = np.array([get_perplexity(val_conf + i) for i in laplace_interval])
print '2b', res_2b.min(), 'smoothing =', laplace_interval[res_2b.argmin()]
# 2.c validation confmat, Kneser-ney smoothing
res_2c = get_perplexity(co.Concept.perform_smoothing(val_conf, ('kneserney', 0.75, 1)))
print '2c', res_2c

# cheating: testing confmat
print '3a (cheating)', get_perplexity(test_conf)

approx_file = np.load(ROOT + 'approximate_confmats.npz')
approx_id = 1
approx_conf = approx_file['confmats'][approx_id]
approx_scale = approx_file['scales'][approx_id]
print '4.0 One-step hessian unlearn, scale %f, accu %f' % \
        (approx_scale, get_acc(approx_conf))
# 4.a approximated confmat, no smoothing
print '4.0a', get_perplexity(approx_conf)
# 4.b approximated confmat, laplace smoothing
res_4b = np.array([get_perplexity(approx_conf + i) for i in laplace_interval])
print '4.0b', res_4b.min(), 'smoothing =', laplace_interval[res_4b.argmin()]
# 4.c approximated confmat, Kneser-ney smoothing
res_4c = get_perplexity(co.Concept.perform_smoothing(approx_conf, ('kneserney', 0.75, 1)))
print '4.0c', res_4c

approx_file = np.load(ROOT + 'adagrad_confmats.npz')
approx_id = 1
approx_conf = approx_file['confmats'][approx_id]
approx_scale = approx_file['scales'][approx_id]
print '4.1 One-step adagrad unlearn, scale %f, accu %f' % \
        (approx_scale, get_acc(approx_conf))
# 4.a approximated confmat, no smoothing
print '4.1a', get_perplexity(approx_conf)
# 4.b approximated confmat, laplace smoothing
res_4b = np.array([get_perplexity(approx_conf + i) for i in laplace_interval])
print '4.1b', res_4b.min(), 'smoothing =', laplace_interval[res_4b.argmin()]
# 4.c approximated confmat, Kneser-ney smoothing
res_4c = get_perplexity(co.Concept.perform_smoothing(approx_conf, ('kneserney', 0.75, 1)))
print '4.1c', res_4c

approx_file = np.load(ROOT + 'approximate_confmats_val.npz')
approx_id = 0
approx_conf = approx_file['confmats'][approx_id]
approx_scale = approx_file['scales'][approx_id]
print '4.2 One-step unlearn, class-based match, max scale %f, accu %f' % \
        (approx_scale, get_acc(approx_conf))
# 4.a approximated confmat, no smoothing
print '4.2a', get_perplexity(approx_conf)
# 4.b approximated confmat, laplace smoothing
res_4b = np.array([get_perplexity(approx_conf + i) for i in laplace_interval])
print '4.2b', res_4b.min(), 'smoothing =', laplace_interval[res_4b.argmin()]
# 4.c approximated confmat, Kneser-ney smoothing
res_4c = get_perplexity(co.Concept.perform_smoothing(approx_conf, ('kneserney', 0.75, 1)))
print '4.2c', res_4c

approx_file = np.load(ROOT + 'adagrad_confmats_val.npz')
approx_id = 0
approx_conf = approx_file['confmats'][approx_id]
approx_scale = approx_file['scales'][approx_id]
print '4.3 One-step adagrad, unlearn, class-based match, max scale %f, accu %f' % \
        (approx_scale, get_acc(approx_conf))
# 4.a approximated confmat, no smoothing
print '4.3a', get_perplexity(approx_conf)
# 4.b approximated confmat, laplace smoothing
res_4b = np.array([get_perplexity(approx_conf + i) for i in laplace_interval])
print '4.3b', res_4b.min(), 'smoothing =', laplace_interval[res_4b.argmin()]
# 4.c approximated confmat, Kneser-ney smoothing
res_4c = get_perplexity(co.Concept.perform_smoothing(approx_conf, ('kneserney', 0.75, 1)))
print '4.3c', res_4c

