import cPickle as pickle
from iceberk import mpi, classifier, mathutil
import numpy as np
import logging
import os, sys
import time
import gflags

def forward(X, outputs):
    output = mathutil.dot(X, w)
    output += b

