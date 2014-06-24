from context import *
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import sklearn


class TestBatchClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.set_printoptions(3)
        N = 64000
        K = 4
        batches = 4
        batch_size = N / batches
        D = 8
        X = np.zeros((N, D * batches))
        row_splits = range(0, N + batch_size, batch_size)
        col_splits = range(0, D * batches + D, D)
        Y = []
        for i in range(len(row_splits) - 1):
            x, y = make_classification(
                n_samples=batch_size, n_features=D, n_informative=D - 2,
                n_classes=K)
            X[row_splits[i]:row_splits[i + 1], col_splits[i]:col_splits[i + 1]] = x
            Y.append(y)
        cls.X = X
        cls.y = np.hstack(Y)
        cls.splits = row_splits
        cls.K = K
        cls.classes = np.arange(K)

    def batch(self, clf_string):
        """
        Test taking 1, 2, 3, 4 batches.
        Accuracy should stay low.
        """
        clf = eval(clf_string)
        clf.fit(self.X, self.y)
        max_acc = accuracy_score(self.y, clf.predict(self.X))
        
        clf = eval(clf_string)
        times = []
        accs = []
        for i in range(len(self.splits) - 1):
            X = self.X[slice(*self.splits[i:i + 2]), :]
            y = self.y[slice(*self.splits[i:i + 2])]
            t = time.time()
            clf.fit(X, y)
            times.append(time.time() - t)
            conf = clf.decision_function(self.X)
            y_pred = conf.argmax(axis=1)
            accs.append(accuracy_score(y_pred, self.y))
        
        print('\n-----------------------')
        print('batch')
        print(clf)
        print('max_acc: {}'.format(max_acc))
        print('Accs:\t{}'.format(np.array(accs)))
        print('Times:\t{}'.format(np.array(times)))

    def sumBatch(self, clf_string):
        """
        Test taking 1, (1,2), (1,2,3), (1,2,3,4) batches.
        Should lead to high accuracy.
        """
        clf = eval(clf_string)
        clf.fit(self.X, self.y)
        max_acc = accuracy_score(self.y, clf.predict(self.X))

        clf = eval(clf_string)
        times = []
        accs = []
        for i in range(len(self.splits) - 1):
            X = self.X[slice(self.splits[0], self.splits[i + 1]), :]
            y = self.y[slice(self.splits[0], self.splits[i + 1])]
            t = time.time()
            clf.fit(X, y)
            times.append(time.time() - t)
            conf = clf.decision_function(self.X)
            y_pred = conf.argmax(axis=1)
            accs.append(accuracy_score(y_pred, self.y))
        
        print('\n-----------------------')
        print('sum batch')
        print(clf)
        print('max_acc: {}'.format(max_acc))
        print('Accs:\t{}'.format(np.array(accs)))
        print('Times:\t{}'.format(np.array(times)))

    def testLogisticBatch(self):
        self.batch('sklearn.linear_model.LogisticRegression(fit_intercept=False)')

    def testLogisticSumBatch(self):
        self.sumBatch('sklearn.linear_model.LogisticRegression(fit_intercept=False)')

    def testSGDWarmBatch(self):
        self.batch('sklearn.linear_model.SGDClassifier(n_jobs=1, alpha=.1, loss="hinge", n_iter=20, fit_intercept=False, shuffle=True, warm_start=True)')

    def testSGDWarmSumBatch(self):
        self.sumBatch('sklearn.linear_model.SGDClassifier(n_jobs=1, alpha=.1, loss="hinge", n_iter=20, fit_intercept=False, shuffle=True, warm_start=True)')

if __name__ == '__main__':
    unittest.main()
