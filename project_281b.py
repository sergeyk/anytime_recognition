#!/usr/bin/env python

import h5py
import subprocess
import sklearn.datasets
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.preprocessing
from sklearn.metrics import accuracy_score
from collections import defaultdict
import bottleneck as bn
import pandas
import time
import numpy as np
import matplotlib.pyplot as plt
from pymc.distributions import mv_normal_cov_like
import scipy.cluster
import scipy.sparse.linalg
from sklearn.metrics import euclidean_distances
import cPickle as pickle
import os
import sys
import glob
import tc

script_fname = os.path.abspath(__file__)
repo_dirname = os.path.dirname(script_fname)


def random_selection(N, F, budget, num_blocks=-1, seed=0):
    """
    Return mask of selected features for the budget.

    Parameters
    ----------
    N, F:       int
    budget:     float in [0, 1]
    num_blocks: int, optional
        If < 0 or > F, then num_blocks == F and features are deleted independenly.
        Otherwise, features are deleted in blocks of size F / num_blocks.
    seed:       float, optional
        Seed for numpy.random.

    Returns
    -------
    mask:       (N, F) ndarray of boolean
    """
    np.random.seed(seed)
    if num_blocks > F:
        print("Warning: it does not make sense to have more blocks than features, so setting num_blocks to F.")
    if num_blocks > 0 and num_blocks < F:
        block_size = F / num_blocks
        mask = np.random.rand(N, num_blocks) < budget
        mask = np.repeat(mask, block_size, axis=1)
        r = np.mod(F, num_blocks)
        if r > 0:
            addendum = np.repeat(np.atleast_2d(mask[:, -1]).T, r, axis=1)
            mask = np.hstack((mask, addendum))
    else:
        mask = np.random.rand(N, F) < budget
    assert(mask.shape == (N, F))
    return mask


def clustered_selection(N, F, budget, num_blocks=-1, K=10, seed=0):
    """
    Return mask of selected features for the budget.

    Only K distinct masks are generated.
    """
    umasks = random_selection(K, F, budget, num_blocks, seed)
    mask = np.repeat(umasks, N/K + 1, axis=0)
    mask = mask[np.random.permutation(N)]
    return mask


def train_k_classifiers(X, y, mask, K):
    """
    Parameters
    ----------
    X : (N, F) ndarray of float
    y : (N,) ndarray
    mask : (N, F) ndarray of boolean
    K : int
        If -1, then all unique masks are found.

    Returns
    -------
    mask_clustering : MaskClustering

    clfs : list of K' classifiers
        K' <= K or if K == -1 or K >= UK, K' == UK,
        where UK is the number of unique masks.
    """
    mask_clustering = tc.MaskClustering(K).fit(mask)
    cluster_ind = mask_clustering.predict(mask)

    unique_inds = np.unique(cluster_ind)
    clfs = []
    for ind in unique_inds:
        try:
            Xm = X.copy()
            m = mask_clustering.umasks[mask_clustering.umask_to_cluster_map.index(ind)]
            Xm[:, ~m] = 0
            clf = train_classifier(Xm, y)
        except Exception as e:
            print(e)
            clf = None
        clfs.append(clf)
    return mask_clustering, clfs


def predict_k_classifiers(X, mask, mask_clustering, clfs, full_clf):
    cluster_ind = mask_clustering.predict(mask)
    unique_inds = np.unique(cluster_ind)
    y_pred = np.zeros(X.shape[0])
    for ind in unique_inds:
        clf = clfs[ind]
        if clf is None:
            clf = full_clf
        y_pred[cluster_ind == ind] = clf.predict(X[cluster_ind == ind])
    return y_pred


def train_classifier(X, y):
    logreg = sklearn.linear_model.LogisticRegression(dual=False)
    t = time.time()
    cv = 2
    clf = sklearn.grid_search.GridSearchCV(
        estimator=logreg, scoring='accuracy',
        param_grid=[{'C': [.01, 1, 10]}], cv=cv,
        n_jobs=1, verbose=0)
    clf.fit(X, y)
    frac_zeros = float((X==0).sum()) / np.prod(X.shape)
    print('Trained clf with {:.3f} zeros in {:.3f} s'.format(
        frac_zeros, time.time() - t))
    print('Best params: {}'.format(clf.best_params_))
    return clf.best_estimator_


def conditional_gaussian(Xm, mask, S, with_variance=False):
    for i in xrange(Xm.shape[0]):
        obs_ind = mask[i, :]
        if obs_ind.sum() == 0:
            Xm[i, ~obs_ind] = 0
        elif (~obs_ind).sum() == 0:
            continue
        else:
            A = S[np.ix_(obs_ind, obs_ind)]
            C_T = S[np.ix_(~obs_ind, obs_ind)]
            ctainv = np.dot(C_T, np.linalg.pinv(A))
            mean = np.dot(ctainv, Xm[i, obs_ind])
            if with_variance:
                B = S[np.ix_(~obs_ind, ~obs_ind)]
                C = S[np.ix_(obs_ind, ~obs_ind)]
                cov = B - np.dot(ctainv, C)
                Xm[i, ~obs_ind] = np.random.multivariate_normal(mean, cov)
            else:
                Xm[i, ~obs_ind] = mean
    return Xm


def svd_impute(X, mask, X_c, mask_c, ranks=[10, 30, 60], cv=3):
    """
    Impute missing values by regressing to eigenvectors.
    Cross-validates the rank parameter over the given grid, with the
    given number of K-folds.

    Parameters
    ----------
    X : (n', f) ndarray
        May be modified.

    mask : (n', f) ndarray of boolean

    X_c : (n, f) ndarray
        Training data with the same distribution as X.

    mask_c : (n, f) ndarray of boolean
    """
    def _svd_impute(Xm, mask, V):
        for i in xrange(Xm.shape[0]):
            obs = mask[i, :]
            Vo = V[:, obs]
            Vu = V[:, ~obs]
            xo = Xm[i, obs]
            w = np.dot(np.linalg.pinv(np.dot(Vo, Vo.T)), np.dot(Vo, xo))
            Xm[i, ~obs] = np.dot(w, Vu)
        return Xm

    def _svd(X, k=-1, verbose=False):
        t = time.time()
        #_, _, V = scipy.sparse.linalg.svds(X_c[train_ind, :], k)
        _, _, V = scipy.linalg.svd(X_c[train_ind, :])
        del _
        if verbose:
            print('SVD took {:.3f} s'.format(time.time() - t))
        return V

    assert(ranks[-2] < X_c.shape[1] and ranks[-1] < X_c.shape[1])
    if ranks[-1] == -1:
        ranks[-1] = X_c.shape[1] - 1
    max_r = ranks[-1]

    n_folds = 2
    kf = sklearn.cross_validation.KFold(X_c.shape[0], n_folds)
    rmses = np.zeros((n_folds, len(ranks)))
    k = 0
    for train_ind, test_ind in kf:
        X_cm_gt = X_c[test_ind, :].copy()
        X_cm = X_c[test_ind, :].copy()
        mask_cm = mask_c[test_ind, :]
        V = _svd(X_c[train_ind, :], max_r, verbose=(k==0))
        for i, r in enumerate(ranks):
            X_cm = _svd_impute(X_cm, mask_cm, V[:r, :])
            rmses[k, i] = np.sqrt(np.power(X_cm_gt - X_cm, 2).sum())
        k += 1
    best_r = ranks[rmses.mean(0).argmax()]
    print('Best SVD rank: {}'.format(best_r))

    # Now do SVD on the whole training set and impute on train and test
    V = _svd(X_c, best_r, verbose=True)
    X_cm = _svd_impute(X_c.copy(), mask_c, V[:best_r, :])
    Xm = _svd_impute(X, mask, V[:best_r, :])
    return Xm, X_cm


def knn_impute(X, mask, Xc, maskc, yc, how='dot'):
    """
    Note that K is cross-validated based on prediction accuracy,
    not reconstruction error.
    """
    def _knn_dot(X, mask, Xc, k, verbose=False):
        t = time.time()
        nn_ind = np.zeros((X.shape[0], k), dtype=int)
        for n in xrange(X.shape[0]):
            dists = -np.dot(Xc[:, mask[n, :]], X[n, mask[n, :]])
            nn_ind[n, :] = bn.argpartsort(dists, k)[:k]
        if verbose:
            print('Finished knn in {:.3f} s'.format(time.time() - t))
        return nn_ind

    def _knn_euclidean(X, mask, Xc, k, verbose=False):
        t = time.time()
        nn_ind = np.zeros((X.shape[0], k), dtype=int)
        for n in xrange(X.shape[0]):
            dists = euclidean_distances(Xc[:, mask[n, :]], X[n, mask[n, :]], squared=True).flatten()
            nn_ind[n, :] = bn.argpartsort(dists, k)[:k]
        if verbose:
            print('Finished knn in {:.3f} s'.format(time.time() - t))
        return nn_ind

    def _predict(y, nn_ind):
        arr = y[nn_ind]
        axis = 1
        u, indices = np.unique(arr, return_inverse=True)
        return u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(arr.shape),
               None, np.max(indices) + 1), axis=axis)]

    if how == 'dot':
        _knn = _knn_dot
    elif how == 'euclidean':
        _knn = _knn_euclidean
    else:
        raise Exception('Unknown mode')

    # ks = [15]
    # n_folds = 2
    # kf = sklearn.cross_validation.KFold(Xc.shape[0], n_folds)
    # accuracies = np.zeros((n_folds, len(ks)))
    # mses = np.zeros((n_folds, len(ks)))
    # i = 0
    # for train_ind, test_ind in kf:
    #     fold_Xm_gt = Xc[test_ind, :].copy()
    #     fold_Xm = Xc[test_ind, :].copy()
    #     fold_mask = maskc[test_ind, :]
    #     fold_Xc = Xc[train_ind, :]

    #     nn_ind = _knn(fold_Xm, fold_mask, fold_Xc, ks[-1])
    #     for j, k in enumerate(ks):
    #         accuracies[i, j] = sklearn.metrics.accuracy_score(_predict(yc[train_ind], nn_ind[:, :k]), yc[test_ind])
    #         for n in xrange(fold_Xm.shape[0]):
    #             fold_Xm[n, ~fold_mask[n]] = np.mean(fold_Xc[nn_ind[n, :k], :], axis=0)[~fold_mask[n]]
    #         mses[i, j] = np.power(fold_Xm_gt - fold_Xm, 2).sum()
    #     i += 1

    # print('accs: {}'.format(accuracies.mean(0)))
    # print('mses: {}'.format(mses.mean(0)))

    # TODO
    best_k_for_acc = 15  # ks[accuracies.mean(0).argmax()]
    best_k_for_mse = 15  # ks[mses.mean(0).argmin()]
    print('{} NN: best K for acc/mse: {}/{}'.format(how, best_k_for_acc, best_k_for_mse))

    nn_ind = _knn(X, mask, Xc, best_k_for_acc, verbose=True)
    y_pred = _predict(yc, nn_ind)

    #nn_ind = _knn(X, mask, Xc, best_k_for_mse)
    for i in xrange(X.shape[0]):
        X[i, ~mask[i]] = np.mean(Xc[nn_ind[i], :], axis=0)[~mask[i]]

    return X, y_pred


def gaussian_predict(Xm, mask, X, y, labels):
    label_log_probs = [np.log((y == label).sum() / float(len(y))) for label in labels]

    S_per_label = []
    mus = []
    for i, label in enumerate(labels):
        S = np.cov(X[y == label, :].T)
        S += np.eye(S.shape[0]) * 1e-8
        S_per_label.append(S)
        mus.append(X[y == label, :].mean(1))

    probs = np.zeros((Xm.shape[0], len(labels)))
    for i in xrange(Xm.shape[0]):
        obs_ind = mask[i, :]
        if obs_ind.sum() > 0:
            for j, label in enumerate(labels):
                A = S_per_label[j][np.ix_(obs_ind, obs_ind)]
                ll = mv_normal_cov_like(Xm[i, obs_ind], mus[j][obs_ind], A)
                probs[i, j] = np.exp(ll + label_log_probs[j])
        else:
            probs[i, :] = np.exp(label_log_probs)
        probs[i, :] /= probs[i, :].sum()
    y_pred = probs.argmax(1)
    return y_pred


def test_missing_value_methods_for_budget(budget, dataset_name, policy_name, num_blocks):
    """
    Parameters
    ----------
    policy_name : string in ['random', 'clustered']
    """
    print('#####\nBudget: {} '.format(budget))

    data = load_dataset(dataset_name)
    dataset_dirname = repo_dirname + '/281b/' + dataset_name
    clf_full_filename = dataset_dirname + '/clf_full.pickle'
    clf_full = pickle.load(open(clf_full_filename))

    res_dirname = '{}/{}_{}'.format(dataset_dirname, policy_name, num_blocks)

    def save(rmse_res, err_res):
        pickle.dump(rmse_res, open(res_dirname + '/{}_rmse_res.pickle'.format(budget), 'w'), protocol=2)
        pickle.dump(err_res, open(res_dirname + '/{}_err_res.pickle'.format(budget), 'w'), protocol=2)

    rmse_res = defaultdict(dict)
    err_res = defaultdict(dict)

    if policy_name == 'random':
        selection_fn = random_selection
    elif policy_name == 'clustered':
        selection_fn = clustered_selection
    else:
        raise Exception('policy_name does not match a function!')

    X = data['X']
    y = data['y']
    X_test = data['X_test']
    y_test = data['y_test']
    N, F = X.shape
    N_test, F = X_test.shape
    mask = selection_fn(N, F, budget, num_blocks=num_blocks, seed=random_seed)
    mask_test = selection_fn(N_test, F, budget, num_blocks=num_blocks, seed=random_seed)

    # make copies that can be modified, for filling in values
    Xm = data['X'].copy()
    Xm_test = data['X_test'].copy()

    # Mean fill, full
    t = time.time()
    Xm_test[~mask_test] = 0
    rmse = np.sqrt(np.power(X_test - Xm_test, 2).sum())
    rmse_res[budget]['mean'] = {'rmse': rmse, 'time': time.time() - t}

    t = time.time()
    err = 1 - accuracy_score(y_test, clf_full.predict(Xm_test))
    err_res[budget]['mean, full'] = {'err': err, 'time': time.time() - t}

    if budget == 0 or budget == 1:
        save(rmse_res, err_res)
        return rmse_res, err_res

    if True:
        # Mean fill, retrained
        Xm[~mask] = 0
        t = time.time()
        clf = train_classifier(Xm, data['y'])
        err = 1 - accuracy_score(y_test, clf.predict(Xm_test))
        err_res[budget]['mean, retrained'] = {'err': err, 'time': time.time() - t}

    if True and policy_name == 'clustered':
        print('Clustered classifiers, K = 5')
        t = time.time()
        K = 5
        mask_clustering, clfs = train_k_classifiers(X, data['y'], mask, K)
        y_pred = predict_k_classifiers(Xm_test, mask_test, mask_clustering, clfs, clf_full)
        err = 1 - accuracy_score(y_test, y_pred)
        err_res[budget]['mean, retrained, 5 clusters'] = {'err': err, 'time': time.time() - t}

    if True and policy_name == 'clustered':
        print('Clustered classifiers, K = -1')
        t = time.time()
        K = -1
        mask_clustering, clfs = train_k_classifiers(X, data['y'], mask, K)
        y_pred = predict_k_classifiers(Xm_test, mask_test, mask_clustering, clfs, clf_full)
        err = 1 - accuracy_score(y_test, y_pred)
        err_res[budget]['mean, retrained, all (10) clusters'] = {'err': err, 'time': time.time() - t}

    if False:
        print('SVD imputation, full')
        t = time.time()
        Xm_test, Xm = svd_impute(Xm_test, mask_test, X, mask)
        rmse = np.sqrt(np.power(X_test - Xm_test, 2).sum())
        rmse_res[budget]['svd'] = {'rmse': rmse, 'time': time.time() - t}

        t = time.time()
        err = 1 - accuracy_score(y_test, clf_full.predict(Xm_test))
        err_res[budget]['svd, full'] = {'err': err, 'time': time.time() - t}

        # t = time.time()
        # clf = train_classifier(Xm, data['y'])
        # err = 1 - accuracy_score(y_test, clf.predict(Xm_test))
        # err_res[budget]['svd, retrained'] = {'err': err, 'time': time.time() - t}

    if True:
        print('Joint Gaussian conditioning on observed elements')
        t = time.time()
        S = np.cov(X.T)
        S += np.eye(S.shape[0]) * 1e-6  # fix singularity
        Xm_test = conditional_gaussian(Xm_test, mask_test, S)

        rmse = np.sqrt(np.power(X_test - Xm_test, 2).sum())
        rmse_res[budget]['gaussian'] = {'rmse': rmse, 'time': time.time() - t}

        t = time.time()
        err = 1 - accuracy_score(y_test, clf_full.predict(Xm_test))
        err_res[budget]['gaussian, full'] = {'err': err, 'time': time.time() - t}

        t = time.time()
        Xm = conditional_gaussian(Xm, mask, S)
        clf = train_classifier(Xm, y)
        err = 1 - accuracy_score(y_test, clf.predict(Xm_test))
        err_res[budget]['gaussian, retrained'] = {'err': err, 'time': time.time() - t}

    # this is worse than even mean imputation!
    if False:
        print('Joint Gaussian conditioning with covariances on observed elements')
        t = time.time()
        Xm_test = conditional_gaussian(Xm_test, mask_test, S, with_variance=True)

        rmse = np.sqrt(np.power(X_test - Xm_test, 2).sum())
        rmse_res[budget]['gaussian w/ cov'] = {'rmse': rmse, 'time': time.time() - t}

        t = time.time()
        err = 1 - accuracy_score(y_test, clf_full.predict(Xm_test))
        err_res[budget]['gaussian w/ cov, full'] = {'err': err, 'time': time.time() - t}

    if True:
        print('kNN dot product')
        t = time.time()
        Xm_test, y_pred = knn_impute(Xm_test, mask_test, X, mask, y, 'dot')
        rmse = np.sqrt(np.power(data['X_test'] - Xm_test, 2).sum())
        err = 1 - accuracy_score(y_test, y_pred)
        rmse_res[budget]['kNN (dot)'] = {'rmse': rmse, 'err': err, 'time': time.time() - t}
        err_res[budget]['kNN (dot)'] = {'err': err, 'time': time.time() - t}

        t = time.time()
        err = 1 - accuracy_score(y_test, clf_full.predict(Xm_test))
        err_res[budget]['kNN (dot), full'] = {'err': err, 'time': time.time() - t}

    if True:
        print('kNN Euclidean')
        t = time.time()
        Xm_test, y_pred = knn_impute(Xm_test, mask_test, X, mask, y, 'euclidean')
        rmse = np.sqrt(np.power(X_test - Xm_test, 2).sum())
        err = 1 - accuracy_score(y_test, y_pred)
        rmse_res[budget]['kNN (euclidean)'] = {'rmse': rmse, 'time': time.time() - t}
        err_res[budget]['kNN (euclidean)'] = {'err': err, 'time': time.time() - t}

        t = time.time()
        err = 1 - accuracy_score(y_test, clf_full.predict(Xm_test))
        err_res[budget]['kNN (euclidean), full'] = {'err': err, 'time': time.time() - t}

    save(rmse_res, err_res)
    return rmse_res, err_res


def process_sklearn_data(dataset, standardize=True, times=1):
    X, X_test, y, y_test = sklearn.cross_validation.train_test_split(
        dataset['data'], dataset['target'], test_size=0.33, random_state=42)
    if times > 1:
        X = np.tile(X, (times, 1))
        y = np.tile(y, (times, 1)).flatten()
    if standardize:
        ss = sklearn.preprocessing.StandardScaler()
        X = ss.fit_transform(X)
        X_test = ss.transform(X_test)
    labels = np.unique(dataset['target'])
    data = locals()
    del data['dataset']
    return data


def load_dataset(dataset_name, standardize=True):
    if dataset_name == 'digits':
        data = process_sklearn_data(sklearn.datasets.load_digits(), standardize, times=2)
    elif dataset_name == 'mnist':
        d = h5py.File(repo_dirname + '/ext/mcf/data/small_mnist_train.mat')
        dt = h5py.File(repo_dirname + '/ext/mcf/data/small_mnist_test.mat')
        data = {
            'X': np.array(d['train_X'], dtype=float),
            'y': np.array(d['train_labels'], dtype=int).flatten(),
            'X_test': np.array(dt['test_X'], dtype=float),
            'y_test': np.array(dt['test_labels'], dtype=int).flatten(),
        }
        if standardize:
            data['ss'] = sklearn.preprocessing.StandardScaler()
            data['X'] = data['ss'].fit_transform(data['X'])
            data['X_test'] = data['ss'].transform(data['X_test'])
        data['labels'] = np.unique(data['y'])
    elif dataset_name == 'scenes':
        ds = tc.data_sources.Scene15()
        data = ds.__dict__
    else:
        raise Exception('Which dataset?')
    return data


def load_results(dirname):
    rmse_res = {}
    for filename in glob.glob(dirname + '/*_rmse_res.pickle'):
        r = pickle.load(open(filename))
        rmse_res.update(r)
    err_res = {}
    for filename in glob.glob(dirname + '/*_err_res.pickle'):
        r = pickle.load(open(filename))
        err_res.update(r)

    rmse_panel = pandas.Panel.from_dict(rmse_res)
    rmse_panel.loc[0.0, 'rmse', :] = rmse_panel.loc[0.0, 'rmse', 'mean']
    rmse_panel.loc[1.0, 'rmse', :] = rmse_panel.loc[1.0, 'rmse', 'mean']

    err_panel = pandas.Panel.from_dict(err_res)
    err_panel.loc[0.0, 'err', :] = err_panel.loc[0.0, 'err', 'mean, full']
    err_panel.loc[1.0, 'err', :] = err_panel.loc[1.0, 'err', 'mean, full']

    return rmse_panel, err_panel


def run_experiment(dataset_name, policy_name, num_blocks, parallel=False):
    # Dataset
    dataset_dirname = repo_dirname + '/281b/' + dataset_name
    if not os.path.exists(dataset_dirname):
        os.mkdir(dataset_dirname)
    data = load_dataset(dataset_name)

    clf_full_filename = dataset_dirname + '/clf_full.pickle'
    if not os.path.exists(clf_full_filename):
        clf_full = train_classifier(data['X'], data['y'])
        pickle.dump(clf_full, open(clf_full_filename, 'w'), protocol=2)

    # Policy
    res_dirname = '{}/{}_{}'.format(dataset_dirname, policy_name, num_blocks)
    if not os.path.exists(res_dirname):
        os.mkdir(res_dirname)

    # Budgets
    budgets = [0, .2, .4, .6, .8, 1]
    if dataset_name == 'scenes':
        budgets = [0, .1, .2, .3, .4, .6, .8, 1]

    if parallel:
        ps = [subprocess.Popen('{} {} {} {} {}'.format(
            script_fname, budget, dataset_name, policy_name, num_blocks), shell=True)
            for budget in budgets]
        for p in ps:
            p.communicate()
    else:
        for budget in budgets:
            test_missing_value_methods_for_budget(budget, dataset_name, policy_name, num_blocks)


def plot_results(dataset_name, policy_name, num_blocks):
    res_dirname = '281b/{}/{}_{}'.format(dataset_name, policy_name, num_blocks)

    rmse_panel, err_panel = load_results(res_dirname)

    err_df = err_panel.loc[:, 'err', :].T
    rmse_df = rmse_panel.loc[:, 'rmse', :].T

    # adding mcf results for one of the experimental settings
    if policy_name == 'random' and num_blocks == -1:
        if dataset_name == 'digits':
            err_df = err_df.join(pandas.DataFrame(
                data=[.9074, .5859, .3485, .2290, .1330, .1077],
                index=[0, .2, .4, .6, .8, 1], columns=['mcf quad']))
            err_df = err_df.join(pandas.DataFrame(
                data=[.9125, .5943, .3418, .2121, .1111, .0892],
                index=[0, .2, .4, .6, .8, 1], columns=['mcf log']))
        elif dataset_name == 'scenes':
            err_df = err_df.join(pandas.DataFrame(
                data=[.9050, .7456, .6741, .5515, .5181, .3561, .2236, .1499],
                index=[0, .1, .2, .3, .4, .6, .8, 1], columns=['mcf quad']))
            err_df = err_df.join(pandas.DataFrame(
                data=[0.9197, .6439, 0.4418, .3440, 0.2972, 0.2035, 0.1406, 0.1312],
                index=[0, .1, .2, .3, .4, .6, .8, 1], columns=['mcf log']))

    # for convenient calculating of AUC
    auc = lambda s: sklearn.metrics.auc(s.index, s.values)
    rmse_df.columns = ['{}: {:.3f}'.format(c, auc(rmse_df[c])) for c in rmse_df.columns]
    err_df.columns = ['{}: {:.3f}'.format(c, auc(err_df[c])) for c in err_df.columns]

    print rmse_df.columns
    print err_df.columns

    cols = 5
    figsize = (30, 5)
    if policy_name == 'random':
        cols = 4
        figsize = (25, 5)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1, cols, 1)
    ax = rmse_df.filter(regex='mean|gaussian|kNN').plot(marker='s', ax=ax)
    ax.set_xlabel('Budget')
    ax.set_ylabel('RMSE')
    ax.set_title('Reconstruction Error vs. Budget')

    ax = fig.add_subplot(1, cols, 2)
    rmse_panel.loc[:, 'time', :].T.filter(regex='mean|gaussian|kNN').plot(marker='s', ax=ax)
    ax.set_xlabel('Budget')
    ax.set_ylabel('Imputation Time')
    ax.set_title('Imputation Time vs. Budget')

    ax = fig.add_subplot(1, cols, 3)
    err_df.filter(regex='mean, retrained:|gaussian, retrained:|kNN \(euclidean\):|kNN \(dot\):|mcf').plot(marker='s', ax=ax)
    ax.set_xlabel('Budget')
    ax.set_ylabel('Classification Error')
    ax.set_title('Classification Error vs. Budget: Best Approaches')

    ax = fig.add_subplot(1, cols, 4)
    err_df.filter(regex='mean, full:|mean, retrained:|gaussian, full:|gaussian, retrained:|kNN \(euclidean\)').plot(marker='s', ax=ax)
    ax.set_xlabel('Budget')
    ax.set_ylabel('Classification Error')
    ax.set_title('Classification Error vs. Budget: Retraining')

    if cols > 4:
        ax = fig.add_subplot(1, cols, 5)
        err_df.filter(regex='mean, retrained').plot(marker='s', ax=ax)
        ax.set_xlabel('Budget')
        ax.set_ylabel('Classification Error')
        ax.set_title('Classification Error vs. Budget: Clustering Classifiers')

    plt.tight_layout()

    plt.savefig(res_dirname + '/subplots.png', dpi=300)

    return rmse_df, err_df


def output_auc_tables(params):
    auc = lambda s: sklearn.metrics.auc(s.index, s.values)
    rmse_auc_df = pandas.DataFrame()
    err_auc_df = pandas.DataFrame()
    for dataset_name, policy_name, num_blocks in params:
        res_dirname = '281b/{}/{}_{}'.format(dataset_name, policy_name, num_blocks)
        rmse_panel, err_panel = load_results(res_dirname)
        rmse_df = rmse_panel.loc[:, 'rmse', :].T
        err_df = err_panel.loc[:, 'err', :].T

        name = policy_name
        if num_blocks > 0:
            name += ', blocks'

        data = dict([(c, auc(rmse_df[c])) for c in rmse_df.columns])
        rmse_auc_df = rmse_auc_df.append(pandas.DataFrame(data, index=[name]))

        data = dict([(c, auc(err_df[c])) for c in err_df.columns])
        err_auc_df = err_auc_df.append(pandas.DataFrame(data, index=[name]))

    rmse_auc_table = rmse_auc_df.T.to_latex(float_format=lambda x: '{:.2f}'.format(x))
    with open('281b/{}/rmse_auc_table.tex'.format(dataset_name), 'w') as f:
        f.write(rmse_auc_table)
    print rmse_auc_table

    err_auc_table = err_auc_df.T.to_latex(float_format=lambda x: '{:.3f}'.format(x))
    with open('281b/{}/err_auc_table.tex'.format(dataset_name), 'w') as f:
        f.write(err_auc_table)
    print err_auc_table


if __name__ == '__main__':
    random_seed = 42
    np.random.seed(random_seed)

    # This allows this same script to be called with command line arguments,
    # which is used for parallelization of computation.
    if len(sys.argv) > 1:
        budget = float(sys.argv[1])
        dataset_name = sys.argv[2]
        policy_name = sys.argv[3]
        num_blocks = int(sys.argv[4])
        test_missing_value_methods_for_budget(
            budget, dataset_name, policy_name, num_blocks)
        sys.exit(0)

    params = [
        #('digits', 'random', -1),
        #('digits', 'random', 8),
        ('digits', 'clustered', -1),
        ('digits', 'clustered', 8),
        ('scenes', 'clustered', -1),
        ('scenes', 'clustered', 5)
    ]

    # params = [
    #     ('scenes', 'random', -1),
    #     ('scenes', 'random', 5),
    #     ('scenes', 'clustered', -1),
    #     ('scenes', 'clustered', 5)
    # ]

    parallel = True
    for dataset_name, policy_name, num_blocks in params:
        run_experiment(dataset_name, policy_name, num_blocks, parallel)
        plot_results(dataset_name, policy_name, num_blocks)
    output_auc_tables(params)
