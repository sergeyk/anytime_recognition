import cPickle as pickle
import glob
import logging
import numpy as np
import os
import socket

# settings
# the directory that contains the images
FEATURE_DIR = "/tscratch/tmp/jiayq/imagenet-vanilla-collected/"
# the mapreduce output
MAPREDUCE_OUTPUT = "mr_extract_vanilla_coates_features.pickle"
# the output mat file
FEATURE_INPUT = "*feature.mmap"
META_INPUT = "*meta.pickle"
CHUNKSIZE = 10000

FEATURE_OUTPUT_PREFIX = FEATURE_DIR + "Xtrain"
LABEL_OUTPUT = FEATURE_DIR + "Ytrain.npy"
LABEL_LIST = FEATURE_DIR + "labels.txt"

def chop():
    feat_files = glob.glob(os.path.join(FEATURE_DIR, FEATURE_INPUT))
    meta_files = glob.glob(os.path.join(FEATURE_DIR, META_INPUT))
    feat_files.sort()
    meta_files.sort()
    meta = [pickle.load(open(f)) for f in meta_files]
    names = sum([m[1] for m in meta], [])
    labels_str = [n[:n.find('_')] for n in names]
    labellist = list(set(labels_str))
    labellist.sort()
    with open(LABEL_LIST, 'w') as fid:
        fid.write('\n'.join(labellist))
    label_to_id = dict([(l,i) for i,l in enumerate(labellist)])
    labels = np.array([label_to_id[label] for label in labels_str])
    np.save(LABEL_OUTPUT, labels)
    features = [np.memmap(f, np.float64, 'r', shape=m[0]) \
            for f,m in zip(feat_files, meta)]
    chunk = np.empty((CHUNKSIZE,) + meta[0][0][1:])
    batch_idx = 0
    chunk_idx = 0
    logging.info("Processing...")
    for i in range(len(features)):
        feat = features[i]
        feat_idx = 0
        feat_size = feat.shape[0]
        while feat_idx < feat_size:
            # fill the current chunk using the current feat
            n = min(feat_size - feat_idx, CHUNKSIZE - chunk_idx)
            chunk[chunk_idx:chunk_idx + n] = feat[feat_idx:feat_idx + n]
            chunk_idx += n
            feat_idx += n
            if chunk_idx == CHUNKSIZE:
                filename = FEATURE_OUTPUT_PREFIX + "-%05d.npy" % batch_idx
                logging.info("Saving chunk %d" % batch_idx)
                np.save(filename, chunk)
                batch_idx += 1
                chunk_idx = 0
        # I have depleted this part, close it to save space
        features[i] = None
        del feat
    # saving the last chunk
    if chunk_idx > 0:
        filename = FEATURE_OUTPUT_PREFIX + "-%05d.npy" % batch_idx
        logging.info("Saving chunk %d" % batch_idx)
        np.save(filename, chunk[:chunk_idx])
        batch_idx += 1
    # now, rename all the files so they have the right name
    for i in range(batch_idx):
        os.rename(FEATURE_OUTPUT_PREFIX + "-%05d.npy" % i,
                  FEATURE_OUTPUT_PREFIX + "-%05d-of-%05d.npy" % (i, batch_idx))
    return

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    chop()
