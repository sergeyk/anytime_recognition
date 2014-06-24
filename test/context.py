import sys
import os
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_dir)
support_dir = os.path.join(repo_dir, 'test/support')

from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import *
import unittest
import time

import tc
