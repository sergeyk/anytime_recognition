"""This script generates the imagenet trials that we can run on AMT to test
human generalization behaviors. The data is built on ILSVRC 2010 test images.
"""
from birdmix import tax
from collections import defaultdict
import numpy as np
import random

############################
# method definition
############################
def get_parent_path(graph, leaf):
    parents = [leaf]
    while len(graph.predecessors(parents[-1])) > 0:
        parents.append(graph.predecessors(parents[-1])[0])
    return parents

def get_trial_sets(graph, leaves, diff = 2):
    """This function gets the trial sets for each leaf node in this graph.
    The trial sets is a list of 5 lists:
        leaf itself
        10% to root leaves
        25% to root leaves
        50% to root leaves
        all other leaves
    """
    trialsets = {}
    for leaf in leaves:
        parents = get_parent_path(graph, leaf)
        psizes = [len(graph.node[p]['leaves']) for p in parents]
        root = parents[-1]

        l1id = 1
        while l1id < len(parents) -1 and psizes[l1id] < 5:
            l1id += 1
        l2id = min(l1id + 1, len(parents) - 1)
        while l2id < len(parents)-1 and \
              (psizes[l2id] == psizes[l1id] or psizes[l2id] < 50):
            l2id += 1
        l3id = min(l2id + 1, len(parents) - 1)
        while l3id < len(parents)-1 and \
              (psizes[l3id] == psizes[l2id] or psizes[l3id] < 200):
            l3id += 1
        trialset = [[leaf],
               list(graph.node[parents[l1id]]['leaves']),
               list(graph.node[parents[l2id]]['leaves']),
               list(graph.node[parents[l3id]]['leaves']),
               list(graph.node[root]['leaves'])]
        """
        p10id = min(int(len(parents) / 10. + 0.5), len(parents)-1)
        p10 = parents[p10id]
        while (p10 != root and len(graph.node[p10]['leaves']) < diff + 1):
               p10id += 1
               p10 = parents[p10id]
        p25id = min(max(p10id+1, int(len(parents) / 4. + 0.5)), len(parents)-1)
        p25 = parents[p25id]
        while (p25 != root and len(graph.node[p25]['leaves']) < len(graph.node[p10]['leaves']) + diff):
               p25id += 1
               p25 = parents[p25id]
        p50id = min(max(p25id+1, int(len(parents) / 2. + 0.5)), len(parents)-1)
        p50 = parents[p50id]
        while (p50 != root and len(graph.node[p50]['leaves']) < len(graph.node[p25]['leaves']) + diff):
               p50id += 1
               p50 = parents[p50id]
        trialset = [[leaf],
               graph.node[p10]['leaves'],
               graph.node[p25]['leaves'],
               graph.node[p50]['leaves'],
               graph.node[root]['leaves']]
        """
        #for i in range(4,0,-1):
        #    trialset[i] = list(set(trialset[i]).difference(trialset[i-1]))
        trialsets[leaf] = trialset
    return trialsets

def generate_trial(trialset, synset2img, trialtype, num_imgs):
    """generate a trial from the given trialset and image maps
    """
    # randomly shuffle the sets.
    for s in trialset:
        random.shuffle(s)
    source = trialset[trialtype]
    # sample images
    # make sure we have the most specific guy
    src_imgs = [random.choice(synset2img[trialset[0][0]])]
    for i in range(num_imgs - 1):
        synset = random.choice(source)
        src_imgs.append(random.choice(synset2img[synset]))
    target_imgs = []
    # target imgs are sampled in a structured way
    # 12 images in domain
    for i in range(4):
        for j in range(3):
            synset = random.choice(trialset[i])
            target_imgs.append(random.choice(synset2img[synset]))
    # 12 images outside the domain
    for i in range(12):
        synset = random.choice(trialset[-1])
        target_imgs.append(random.choice(synset2img[synset]))
    # shuffling the images to minimize the ordering effect
    random.shuffle(src_imgs)
    random.shuffle(target_imgs)
    return src_imgs, target_imgs

def debug_show_trial(src_imgs, target_imgs, filename):
    template = '<img height=100px src="http://www.icsi.berkeley.edu/~jiayq/ilsvrctest/ILSVRC2010_test_%08d.JPEG" />'
    with open(filename,'w') as fid:
        fid.write('Training images<br/>\n')
        for img in src_imgs:
            fid.write(template % img + '\n')
        fid.write('<br/>Testing images<br/>\n')
        for img in target_imgs:
            fid.write(template % img + '\n')
    return

############################
# method definition Done. Script starts here.
############################

graph = tax.get_imagenet_taxonomy(1000)
tax.compute_leave_sets(graph)
leaves = [n for n in graph.nodes() if len(graph.successors(n)) == 0]
leaves.sort()
test_labels = np.load(\
        '/u/vis/x1/jiayq/ILSVRC/test_predict/labels_ascii_sorted.npy')
test_labels = test_labels.astype(int)
# this structure stores the image indices for test labels
synset2img = defaultdict(list)
for i, v in enumerate(test_labels):
    synset2img[leaves[v]].append(i+1)
trialsets = get_trial_sets(graph, leaves)
