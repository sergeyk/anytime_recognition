"""
Yahoo! Learning to rank challenge.

> data = h5py.File(ltrc_dirname + '/set1.h5')
> for x in ['train', 'valid', 'test']:
>     print data['/{}/X'.format(x)].shape
(473134, 699)
(71083, 699)
(165660, 699)
> print np.unique(data['/test/y'])
> print np.bincount(data['/test/y'])
[ 0.  1.  2.  3.  4.]
[42625 59107 48033 12804  3091]

> data = h5py.File(ltrc_dirname + '/set2.h5')
> for x in ['train', 'valid', 'test']:
>     print data['/{}/X'.format(x)].shape
(34815, 700)
(34881, 700)
(103174, 700)
> print np.unique(data['/test/y'])
> print np.bincount(data['/test/y'])
[ 0.  1.  2.  3.  4.]
[22461 51598 23167  4234  1714]

Out of the 700 features, only 519 have acceptable weight and are actually used.
"""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py
from sklearn.datasets import load_svmlight_file
import tc

ltrc_dirname = tc.repo_dir + '/data/ltrc_yahoo'


def transform_to_h5():
    """
    One-time transformation of the data to h5 format that is faster to load.
    """
    # this took about 10 minutes for set1
    for setname in ['set1', 'set2']:
        filename = os.path.join(ltrc_dirname, '{}.h5'.format(setname))
        f = h5py.File(filename, 'w')

        for name in ['train', 'valid', 'test']:
            g = f.create_group(name)
            filename = os.path.join(ltrc_dirname, '{}.{}.txt'.format(setname, name))
            X, y, q = load_svmlight_file(filename, query_id=True)
            g.create_dataset('X', data=X.todense(), compression='gzip')
            g.create_dataset('y', data=y, compression='gzip')
            g.create_dataset('q', data=q, compression='gzip')
        f.close()
    # Now you can do this
    #     f['/valid/X'].shape
    #     Out[24]: (71083, 699)


class LTRC(tc.DataSource):
    """
    """
    def __init__(self, dirname, max_budget=None):
        self.dirname = dirname

        action_costs = np.loadtxt(ltrc_dirname + '/featurecost.csv', delimiter=',')
        good_ind = action_costs != 99999999
        self.action_costs = action_costs[good_ind]
        self.action_dims = np.ones(len(self.action_costs), dtype=int)
        self.actions = np.arange(len(self.action_costs)).tolist()
        self.labels = np.arange(5)

        self.max_budget = max_budget
        if max_budget is None:
            self.max_budget = int(sum(self.action_costs) / 4)

        self.data_filename = self.dirname + '/ltrc_set2.h5'
        if not os.path.exists(self.data_filename):
            data = h5py.File(ltrc_dirname + '/set2.h5')
            X = data['/train/X'][:, good_ind]
            y = data['/train/y']
            X_test = data['/valid/X'][:, good_ind]
            y_test = data['/valid/y']

            ss = StandardScaler()
            X = ss.fit_transform(X)
            X_test = ss.transform(X_test)
            with h5py.File(self.data_filename, 'w') as f:
                f.create_dataset('X', data=X)
                f.create_dataset('y', data=y)
                f.create_dataset('X_test', data=X_test)
                f.create_dataset('y_test', data=y_test)
        self.N = self.X.shape[0]
        self.N_test = self.X_test.shape[0]

    @property
    def name(self):
        return 'ltrc_{}'.format(self.max_budget)
