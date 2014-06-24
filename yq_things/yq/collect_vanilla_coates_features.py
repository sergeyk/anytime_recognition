import cPickle as pickle
import logging
import numpy as np
import os
import socket

# settings

# the directory that contains the images
FEATURE_DIR = "/tscratch/tmp/jiayq/imagenet-vanilla/"
# the mapreduce output
MAPREDUCE_OUTPUT = "mr_extract_vanilla_coates_features.pickle"


# the output mat file
FEATURE_OUTPUT = "feature.mmap"
META_OUTPUT = "meta.pickle"

def deal_with_mine():
    dict = pickle.load(open(MAPREDUCE_OUTPUT))
    hostname = socket.gethostname()
    try:
        my_list = dict[hostname]
        del dict
    except KeyError:
        logging.info("I am not responsible for anything. Quitting...")
        return
    if len(my_list) == 0:
        logging.info("I don't have anything to collect. Quitting...")
        return
    logging.info("%s: I have %d images." % (hostname, len(my_list)))
    data = np.load(os.path.join(FEATURE_DIR, my_list[0]+'.npy'))
    feature = np.memmap(
            os.path.join(FEATURE_DIR, hostname + '_' + FEATURE_OUTPUT),
            np.float64, 'w+', shape = (len(my_list),) + data.shape)
    for i, name in enumerate(my_list):
        data = np.load(os.path.join(FEATURE_DIR, name+'.npy'))
        feature[i] = data
    # write metadata
    meta = (feature.shape, my_list)
    pickle.dump(meta, open(
            os.path.join(FEATURE_DIR, hostname + '_' + META_OUTPUT),'w'))
    # finish writing the collected features
    del feature
    logging.info("%s done. %d images." % (hostname, len(my_list)))
    return

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    deal_with_mine()
