import cPickle as pickle
import gflags
from iceberk import datasets
from mincepie import mapreducer, launcher
import numpy as np
import os
import socket

# settings
gflags.DEFINE_string("conv_name", 
        os.path.join(os.path.dirname(__file__), "ilsvrc_vanilla_conv.pickle"),
        "The filename that contains the convolutional network definition.")
gflags.DEFINE_string("image_dir", "",
        "The directory that contains the images.")
gflags.DEFINE_string("feature_dir", "",
        "The directory that we will write features to.")
FLAGS = gflags.FLAGS
# we will crop the center of the image and resize it to 256x256.
CENTER_CROP = 256

class FeatureExtractionMapper(mapreducer.BasicMapper):
    def set_up(self):
        """Load the network, and set up the output path if necessary.
        """
        self._conv = pickle.load(open(FLAGS.conv_name))
        self._hostname = socket.gethostname()
        self._buffer = [None] * (len(self._conv) + 1)
        try:
            os.makedirs(FLAGS.feature_dir)
        except OSError:
            pass

    def read_image(self, name):
        """Reads the image and does the manipulation
        """
        img = datasets.imread_rgb(name)
        return datasets.manipulate(img, None, None, None, CENTER_CROP)

    def map(self, key, value):
        """key will be dummy, and value will be the image filename
        """
        imagename = os.path.basename(value)
        feature = self._conv.process( \
                self.read_image(os.path.join(FLAGS.image_dir, value)),\
                convbuffer = self._buffer)
        np.save(os.path.join(FLAGS.feature_dir, imagename), feature)
        yield self._hostname, imagename
        
mapreducer.REGISTER_DEFAULT_MAPPER(FeatureExtractionMapper)

# for Reduce, we will simply use the identity reducer.
mapreducer.REGISTER_DEFAULT_REDUCER(mapreducer.IdentityReducer)
mapreducer.REGISTER_DEFAULT_READER(mapreducer.FileReader)
mapreducer.REGISTER_DEFAULT_WRITER(mapreducer.PickleWriter)

if __name__ == "__main__":
    launcher.launch()
