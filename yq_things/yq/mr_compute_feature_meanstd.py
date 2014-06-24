import cPickle as pickle
import gflags
from iceberk import datasets
from mincepie import mapreducer, launcher
import numpy as np
import os
import socket

# settings
gflags.DEFINE_string("feature_file", 
        "/tscratch/tmp/jiayq/imagenet-vanilla-collected/Xtrain-%05d-of-00127.npy",
        "The filename template.")
gflags.DEFINE_string("mean", "mean.pickle", "")
gflags.DEFINE_string("std", "std.pickle", "")
gflags.DEFINE_float("reg", 0.01, "")
FLAGS = gflags.FLAGS

class MeanMapper(mapreducer.BasicMapper):
    def map(self, key, value):
        data = np.load(FLAGS.feature_file % key)
        yield "m", (data.sum(0), data.shape[0])

class MeanReducer(mapreducer.BasicReducer):
    def reduce(self, key, values):
        mean = sum(v[0] for v in values) / sum(v[1] for v in values)
        return mean

class StdMapper(mapreducer.BasicMapper):
    def set_up(self):
        self._m = pickle.load(open(FLAGS.mean))['m']

    def map(self, key, value):
        data = np.load(FLAGS.feature_file % key)
        data -= self._m
        yield "std", ((data**2).sum(0), data.shape[0])

class StdReducer(mapreducer.BasicReducer):
    def reduce(self, key, values):
        std = sum(v[0] for v in values) / sum(v[1] for v in values)
        std = np.sqrt(std + FLAGS.reg)
        return std

mapreducer.REGISTER_MAPPER(MeanMapper)
mapreducer.REGISTER_REDUCER(MeanReducer)
mapreducer.REGISTER_MAPPER(StdMapper)
mapreducer.REGISTER_REDUCER(StdReducer)
mapreducer.REGISTER_DEFAULT_READER(mapreducer.IterateReader)
mapreducer.REGISTER_DEFAULT_WRITER(mapreducer.PickleWriter)

if __name__ == "__main__":
    launcher.launch()
