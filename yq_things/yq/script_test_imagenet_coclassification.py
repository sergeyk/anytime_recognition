import cPickle as pickle
import coclassification as co
from birdmix import tax, wordnet
from iceberk import classifier, mathutil
import numpy as np
import logging
import gflags
import random
import os, sys

gflags.DEFINE_string('prior', 'erlang_wn', 'The prior method: erlang_wn, erlang_1k or flat')
gflags.DEFINE_string('prior_gt', 'erlang_wn', 'The prior method to generate the ground truth: erlang_wn, erlang_1k or flat')
gflags.DEFINE_bool('leaf', True, 'If false, we do not set prior for leaf nodes')
gflags.DEFINE_integer('num_sets', 10, 'The number of testing sets')
gflags.DEFINE_integer('set_size', 100, 'The number of images in the set')
gflags.DEFINE_float('class_ratio', 0, 'The ratio of classes to be selected in the task')
gflags.DEFINE_bool('soft', False, 'If set, perform soft prediction')
gflags.DEFINE_float('classifier_weight', 1., 'The weight added to the classifiers')
gflags.DEFINE_integer('seed', -1, 'If positive, set the random seed to generate the test data points.')
gflags.DEFINE_bool('naive', False, 'If set, perform naive baseline')
gflags.DEFINE_bool('oracle', False, 'If set, perform oracle (known task)')
gflags.DEFINE_bool('prototype', False, 'If set, perform prototype baseline')
gflags.DEFINE_bool('histogram', False, 'If set, perform histogram baseline')
gflags.DEFINE_bool('hedging', False, 'If set, perform hedging baseline')
gflags.DEFINE_float('hedging_thres', 0.5, 'The hedging threshold')
gflags.DEFINE_bool('dump', False, 'If set true, dump a html file showing each individual prediction')

print sys.argv
FLAGS = gflags.FLAGS
FLAGS(sys.argv)
WORDNET_LAMBDA = 200 # the erlang parameter for wordnet
ILSVRC_LAMBDA = 25

# first, obtain the prior probability for imagenet
wngraph = wordnet.get_wordnet_hierarchy()
graph = tax.get_imagenet_taxonomy(1000)
tax.compute_leave_sets(graph)
nodelist = graph.nodes()
nodelist.sort()
nodeset = set(nodelist)
node2id = dict([(n,i) for i,n in enumerate(nodelist)])
leaflist = [n for n in graph.nodes() if len(graph.successors(n)) == 0]
leaflist.sort()
leafset = set(leaflist)
leaf2id = dict([(n,i) for i,n in enumerate(leaflist)])

def get_prior(prior_string):
    if prior_string == 'erlang_wn':
        prior = tax.compute_erlang_prior(wngraph, WORDNET_LAMBDA)
        # get only the nodes in the imagenet graph
        prior = dict([(n,prior[n]) for n in prior if n in nodeset])
        # normalize
        total = sum(prior.values())
        for n in prior:
            prior[n] /= total
    elif prior_string == 'flat':
        val = 1. / len(nodelist)
        prior = dict([(n,val) for n in nodelist])
    elif prior_string == 'erlang_1k':
        prior = tax.compute_erlang_prior(graph, ILSVRC_LAMBDA)
    elif prior_string == 'debug':
        prior = dict([(n, 0.) for n in nodelist])
        root = [n for n in nodelist if len(graph.predecessors(n)) == 0]
        prior[root[0]] = 1.
    # check if we want to shut off leaf nodes
    if FLAGS.leaf is False:
        for n in nodelist:
            # set the prior for leaf nodes to be 0.
            if len(graph.node[n]['leaves']) == 1:
                prior[n] = 0.
    # in the end, make the prior a vector
    prior = np.array([prior[n] for n in nodelist])
    return prior

def overlap(va,vb):
    """Computes the overlapping score between two binary vectors va and vb:
        overlap = intersect(a,b) / union(a,b)
    similar to the computaion of overlap in object detection.
    """
    return (va & vb).sum() / float((va | vb).sum())

def getcolor(bool):
    if bool:
        return 'green'
    else:
        return 'red'

prior = get_prior(FLAGS.prior)
prior_gt = get_prior(FLAGS.prior_gt)

# get the conditional probability
conditional = np.zeros((len(nodelist), len(leaflist)))
tax.compute_leave_sets(graph)
for n in nodelist:
    for l in graph.node[n]['leaves']:
        conditional[node2id[n],leaf2id[l]] = 1.
conditional /= conditional.sum(1)[:, np.newaxis]

# get the confusion matrix
conf = np.load('/u/vis/x1/jiayq/ILSVRC/adagrad_confmats_val.npz')
conf = conf['confmats'][0]

# now, construct the concept space
cspace = co.Concept(prior, conditional, conf, smooth=('kneserney', 0.8, 1),
                   graph = graph, concept2id = node2id, leaf2id = leaf2id)
cspace_gt = co.Concept(prior_gt, conditional, conf, smooth=('kneserney', 0.8, 1),
                   graph = graph, concept2id = node2id, leaf2id = leaf2id)
# get the accuracies for the hedging baseline
hedging_accuracies = cspace.hedging_accuracy()

# get the test ground truth labels
test_labels = np.load('/u/vis/x1/jiayq/ILSVRC/test_predict/labels_ascii_sorted.npy')
# set seed with some very simple manipulations
if FLAGS.seed > 0:
    seed = FLAGS.seed
    seed += FLAGS.num_sets * seed * seed + FLAGS.set_size * seed + int(FLAGS.class_ratio * 100)
    random.seed(seed)
    np.random.seed(seed + 1)
concept_gt, label_gt, indices = cspace_gt.generate_testsets(\
        test_labels, FLAGS.num_sets, FLAGS.set_size, FLAGS.class_ratio)

# carry out testing
test_prob = np.load('/u/vis/x1/jiayq/ILSVRC/test_predict/test_prob.npy')
if not FLAGS.soft:
    # convert to hard predictions
    test_pred = test_prob.argmax(1)
    test_prob[:] = 0.
    test_prob[np.arange(test_prob.shape[0]), test_pred] = 1.

# prepare methods
methods = [lambda cid, prob: cspace.coclassify_dag(prob, FLAGS.classifier_weight)]
names = ['Adapt']
if FLAGS.naive:
    methods.append(lambda cid, prob: cspace.coclassify_baseline(prob))
    names.append('Naive')
if FLAGS.oracle:
    methods.append(lambda cid, prob: cspace.coclassify_oracle(prob, cid))
    names.append('Oracle')
if FLAGS.prototype:
    methods.append(lambda cid, prob: cspace.coclassify_prototype(prob))
    names.append('Proto')
if FLAGS.histogram:
    methods.append(lambda cid, prob: cspace.coclassify_histogram(prob))
    names.append('Histo')
if FLAGS.hedging:
    methods.append(lambda cid, prob: \
            cspace.coclassify_hedging(prob, \
                                      FLAGS.hedging_thres, \
                                      hedging_accuracies))
    names.append('H-%.02f' % (FLAGS.hedging_thres))

# record the outputs
cid_distances = np.zeros(len(methods))
cid_overlaps = np.zeros(len(methods))
cid_chisquares = np.zeros(len(methods))
accus = np.zeros(len(methods))

if FLAGS.dump:
    dump_fid = open(__file__+'.html', 'w')
    dump_fid.write('<pre>\n%s\n</pre>\n' % str(sys.argv))
    dump_fid.write('<h1>Test Dump</h1>\n')

for cid, lgt, idx in zip(concept_gt, label_gt, indices):
    if FLAGS.dump:
        #dump_fid.write('<h3>%s</h3>\n' % graph.node[nodelist[cid]]['word'].split(',')[0])
        dump_fid.write('<table border="1">\n')
        dump_fid.write('<tr><td><h3>%s</h3></td>\n' % graph.node[nodelist[cid]]['word'].split(',')[0])
        for i in range(len(lgt)):
            dump_fid.write('<td>%s</td>\n' % graph.node[leaflist[lgt[i]]]['word'].split(',')[0])
        dump_fid.write('</tr>\n<tr><td></td>\n')
        for i in range(len(lgt)):
            dump_fid.write('<td><img width=100 src="/u/vis/x1/common/ILSVRC-2010/test/ILSVRC2010_test_%08d.JPEG" /><br/> (%s)</td>\n' % (idx[i]+1, str(idx[i] + 1)))
        dump_fid.write('</tr>\n')

    for i, method in enumerate(methods):
        cid_pred, label_pred = method(cid, test_prob[idx])
        cid_overlaps[i] += overlap(cspace.membership[cid], cspace.membership[cid_pred])
        cid_distances[i] += (cspace.membership[cid] != cspace.membership[cid_pred]).sum()
        a = cspace.conditional[cid]
        b = cspace.conditional[cid_pred]
        cid_chisquares[i] += ((a-b)**2 / (a+b+np.finfo(np.float64).eps)).sum() / 2
        accus[i] += (label_pred == lgt).sum() / float(len(lgt))
        if FLAGS.dump:
            dump_fid.write('<tr><td>%s<br/>%s</td>\n' % \
                       (names[i], graph.node[nodelist[cid_pred]]['word'].split(',')[0]))
            for i in range(len(lgt)):
                dump_fid.write('<td><font color="%s">%s</font></td>\n' % \
                        (getcolor(lgt[i] == label_pred[i]), \
                         graph.node[leaflist[label_pred[i]]]['word'].split(',')[0]))
            dump_fid.write('</tr>\n')
    if FLAGS.dump:
        dump_fid.write('</table>\n')

N = float(FLAGS.num_sets)
print '%8s %10s %10s %10s %10s' % ('method', 'hamming', 'overlap', 'chi2', 'accu')
for m,d,v,c,a in zip(names, cid_distances, cid_overlaps, cid_chisquares, accus):
    print '%8s %10.04f %10.04f %10.04f %10.04f' % (m, d / N, v / N, c / N, a / N)
