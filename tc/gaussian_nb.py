import numpy as np
import bottleneck as bn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import array2d
from sklearn.utils.extmath import logsumexp
from sklearn.utils import check_arrays


class GaussianNB(BaseEstimator, ClassifierMixin):
    """
    Gaussian Naive Bayes (GaussianNB)

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y : array, shape = [n_samples]
        Target vector relative to X

    Attributes
    ----------
    `class_prior_` : array, shape = [n_classes]
        probability of each class.

    `theta_` : array, shape = [n_classes, n_features]
        mean of each feature per class

    `sigma_` : array, shape = [n_classes, n_features]
        variance of each feature per class

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    def fit(self, X, y, mask=None):
        """Fit Gaussian Naive Bayes according to X, y

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        mask : array-like, shape = [n_samples, n_features]
            Binary, 1 at unobserved features.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_arrays(X, y, sparse_format='dense')

        n_samples, n_features = X.shape

        if n_samples != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")

        if mask is not None:
            mask = array2d(mask)
            X = X.copy()
            X[mask] = np.nan

        self.classes_ = unique_y = np.unique(y)
        n_classes = unique_y.shape[0]

        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        self._n_ij = []
        epsilon = 1e-9
        for i, y_i in enumerate(unique_y):
            self.theta_[i, :] = bn.nanmean(X[y == y_i, :], axis=0)
            self.sigma_[i, :] = bn.nanvar(X[y == y_i, :], axis=0) + epsilon
            self.class_prior_[i] = np.float(np.sum(y == y_i)) / n_samples
            self._n_ij.append(-0.5 * np.sum(np.log(np.pi * self.sigma_[i, :])))
        self._logprior = np.log(self.class_prior_)
        return self

    def _jll(self, X, i):
        n_ij = self._n_ij[i] - 0.5 * bn.nansum(((X - self.theta_[i, :]) ** 2) /
                                              (self.sigma_[i, :]), 1)
        n_ij[np.isnan(n_ij)] = 0
        return self._logprior[i] + n_ij

    def _joint_log_likelihood(self, X, mask=None):
        X = array2d(X)
        if mask is not None:
            mask = array2d(mask)
            X = X.copy()
            X[mask] = np.nan
        joint_log_likelihood = np.zeros((len(self.classes_), X.shape[0]))
        for i in range(np.size(self.classes_)):
            joint_log_likelihood[i, :] = self._jll(X, i)
        return joint_log_likelihood.T

    def predict(self, X, mask=None):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        mask : array-like, shape = [n_samples, n_features]
            Binary, 1 at unobserved features.

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        jll = self._joint_log_likelihood(X, mask)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X, mask=None):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        mask : array-like, shape = [n_samples, n_features]
            Binary, 1 at unobserved features.

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class
            in the model, where classes are ordered arithmetically.
        """
        jll = self._joint_log_likelihood(X, mask)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X, mask=None):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        mask : array-like, shape = [n_samples, n_features]
            Binary, 1 at unobserved features.

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically.
        """
        return np.exp(self.predict_log_proba(X, mask))


if __name__ == '__main__':
    import sklearn.datasets
    data = sklearn.datasets.load_iris()
    X = data['data']
    X = sklearn.preprocessing.StandardScaler().fit_transform(X)
    y = data['target']

    np.random.seed(0)
    mask = np.random.rand(*X.shape) < 0.5
    # mask is 1 at unobserved features
    Xm = X.copy()
    Xm[mask] = 0

    clf = GaussianNB().fit(X, y)

    from sklearn.metrics import accuracy_score
    print('clf.score(X, y): {:.3f}'.format(accuracy_score(clf.predict(X), y)))
    print('clf.score(Xm, y): {:.3f}'.format(accuracy_score(clf.predict(Xm), y)))
    print('clf.score_mask(Xm, y): {:.3f}'.format(accuracy_score(clf.predict(Xm, mask), y)))
