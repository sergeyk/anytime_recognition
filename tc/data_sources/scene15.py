import os
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import operator
from glob import glob
import tc
import h5py
from tc import DataSource


def load_split_info():
    """
    Load the Scene-15 splits into a comprehensible format.
    """
    split_pickle_fname = tc.repo_dir + '/data/scenes15/split10.pickle'
    split_info = tc.util.load_pickle(split_pickle_fname)

    # Pickle file does not exist, must load from .mat
    split_mat_fname = tc.repo_dir + '/data/scenes15/split10.mat'
    data = loadmat(split_mat_fname)
    split_data = [x.flatten() for x in data['split'].flatten()]

    splits = []
    all_classes = []
    for s in xrange(len(split_data)):
        split = []
        classes = []
        for c in xrange(split_data[s].shape[0]):
            classes.append(split_data[s][c]['ClassName'][0][0][0])
            split.append({
                'train_ids': np.array([x[0][0] for x in split_data[s][c]['TrainingID'][0][0][0].flatten()]),
                'test_ids':  np.array([x[0][0] for x in split_data[s][c]['TestingID'][0][0][0].flatten()])
            })
        splits.append(split)
        all_classes.append(classes)
    assert(all([x == all_classes[0] for x in all_classes]))

    split_info = {'classes': all_classes[0], 'splits': splits}
    tc.util.dump_pickle(split_info, split_pickle_fname)
    return split_info


def load_confs():
    """
    Load the confidence matrix for the first split, using the best parameter setting
    of each feature type.

    Results
    -------
    Num feats: 14
    gist:           0.736
    hog2x2:         0.821
    tiny_image:     0.318
    lbp:            0.772
    lbphf:          0.730
    denseSIFT:      0.574
    line_hists:     0.457
    gistPadding:    0.725
    sparse_sift:    0.549
    ssim:           0.793
    texton:         0.810
    geo_map8x8:     0.527
    geo_texton:     0.773
    geo_color:      0.432

    Timing
    ------
    image /MITtallbuilding/image_0097    feature sparse_sift    time 0.734083
    image /MITtallbuilding/image_0097    feature gistPadding    time 0.706929
    image /MITtallbuilding/image_0097    feature line_hists    time 1.195693
    image /MITtallbuilding/image_0097    feature lbp    time 0.367883
    image /MITtallbuilding/image_0097    feature lbphf    time 0.310785
    image /MITtallbuilding/image_0097    feature geo_texton    time 8.105121
    image /MITtallbuilding/image_0097    feature geo_color    time 3.047720
    image /MITtallbuilding/image_0097    feature geo_map8x8    time 2.817572
    image /MITtallbuilding/image_0097    feature hog2x2    time 0.358025
    image /MITtallbuilding/image_0097    feature texton    time 4.114994
    image /MITtallbuilding/image_0097    feature gist    time 0.258822
    image /MITtallbuilding/image_0097    feature tiny_image    time 0.172668
    image /MITtallbuilding/image_0097    feature ssim    time 3.410481
    image /MITtallbuilding/image_0097    feature denseSIFT    time 4.026814
    """
    confs_pickle_fname = tc.repo_dir + '/data/scenes15/confs.pickle'
    if os.path.exists(confs_pickle_fname):
        return tc.util.load_pickle(confs_pickle_fname)

    split_info = load_split_info()
    result_path = tc.repo_dir + '/data/scenes15/result'
    feats = [
        'gist', 'hog2x2', 'tiny_image', 'lbp', 'lbphf',
        'denseSIFT', 'line_hists', 'gistPadding', 'sparse_sift',
        'ssim', 'texton', 'geo_map8x8', 'geo_texton', 'geo_color']
    print('Num feats: {}'.format(len(feats)))
    splits = np.arange(10) + 1

    # find the best parameter setting for each feature type
    filenames = {}
    for feat in feats:
        print('\n'+feat)
        all_accs = []
        for split in splits:
            y = np.array(reduce(operator.add, ([i] * split_info['splits'][0][i]['test_ids'].shape[0] for i in range(15))))
            path = result_path + '/SVM_Result_0100*{}*split_{:02d}*.mat'.format(feat, split)
            accs = []
            for filename in glob(path):
                data = loadmat(filename)  # keys: 'confidence', 'score_test', 'class_hat'
                y_pred = data['score_test'].argmax(1)
                acc = accuracy_score(y, y_pred)
                accs.append(acc)
            all_accs.append(accs)

        all_accs = np.array(all_accs)
        print('{}\n{}'.format(all_accs.mean(0), all_accs.std(0)))
        best_params_ind = all_accs.mean(0).argmax()
        # print('{}:\t{:.3f}'.format(feat, all_accs.mean(0)[best_params_ind]))

        # we are going to use the first split
        path = result_path + '/SVM_Result_0100*{}*split_{:02d}*.mat'.format(feat, 1)
        filename = glob(path)[best_params_ind]
        print('with acc {:.3f} using: {}'.format(all_accs.mean(0)[best_params_ind], filename))
        filenames[feat] = filename

    # assemble the matrix of confidences, using the selected params
    all_confs = []
    for feat in feats:
        confs = loadmat(filenames[feat])['score_test']
        all_confs.append(confs)
    all_confs = np.hstack(all_confs)
    tc.util.dump_pickle(all_confs, confs_pickle_fname)
    return all_confs


class Scene15(DataSource):
    """
    Scene-15 dataset from [1], with classifier outputs as computed by [2].

    [1] S. Lazebnik, C. Schmid, and J. Ponce, "Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories," in CVPR, 2006.
    [2] J. Xiao, J. Hays, K. A. Ehinger, A. Oliva, and A. Torralba, "SUN database: Large-scale scene recognition from abbey to zoo," in CVPR, 2010.
    """
    def __init__(self, dirname, max_budget=None):
        self.dirname = dirname
        self.max_budget = max_budget
        if max_budget is None:
            self.max_budget = 230

        split_info = load_split_info()
        labels = np.array(reduce(operator.add, ([i] * split_info['splits'][0][i]['test_ids'].shape[0] for i in range(15))))
        self.labels = sorted(np.unique(labels))
        self.actions = [
            'gist', 'hog2x2', 'tiny_image', 'lbp', 'lbphf', 'denseSIFT',
            'line_hists', 'gistPadding', 'sparse_sift', 'ssim', 'texton',
            'geo_map8x8', 'geo_texton', 'geo_color'
        ]
        self.action_dims = np.ones(len(self.actions), dtype='int') * 15
        self.action_costs = np.array([
            .259, .358, .173, .368, .311, 4.03,
            1.20, .707, .734, 3.41, 4.12,
            2.81, 8.11, 3.05
        ], dtype=float)

        self.data_filename = self.dirname + '/scene15.h5'
        if not os.path.exists(self.data_filename):
            confs = load_confs()
            X, X_test, y, y_test = train_test_split(
                confs, labels, random_state=42)

            # First, logistic transform to get values on [0,1],
            # then standardize. This gets best performance (by 1%)
            # using all features.
            X = 1. / (1 + np.exp(-X))
            X_test = 1. / (1 + np.exp(-X_test))
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
        return 'scene15_{}'.format(self.max_budget)

if __name__ == '__main__':
    ds = Scene15('data/data_sources')
    ds.save()
