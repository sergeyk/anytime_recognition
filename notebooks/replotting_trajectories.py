
# coding: utf-8

# In[1]:

# Path to experiment to re-plot
dirname = '../data/timely_results/synthetic_orthants_D3_False_N16000_Nt4000_13/linear_untaken-dynamic-1.0-True-logreg-1-False-0-auc-infogain-False-10-0.2-5'


# In[2]:

# Load the stored classifier
from tc.timely_classifier import TimelyClassifier
import cPickle
tc = cPickle.load(open(dirname + '/ticl.pickle'))
print tc
print tc.ds


# In[3]:

# Adjust the filepaths
tc.ds.data_filename = tc.ds.data_filename.replace('/n/banquet/df/sergeyk', '/Users/sergeyk')
tc.logging_dirname = '../' + tc.logging_dirname
tc.report.dirname = '../' + tc.report.dirname
tc.report.html_filename = '../' + tc.report.html_filename


# In[5]:

tc.evaluate(num_workers=4, force=True)

