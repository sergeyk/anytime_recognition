import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import tc


class SyntheticOrthants(tc.DataSource):
    """
    Data are multi-dimensional, generated from Gaussians with either
    class-specific or zero-center (uninformative) means.
    There is a class for each orthant.

    Budget is set to the exact fraction needed by the policy.

    There are two types of features:
    - one for each dimension that returns its sign
    - one for each orthant, which correctly returns the label of the data point
        if it is in the orthant corresponding to the feature, and is otherwise
        uniformly noisy.

    Parameters
    ----------
    dirname: string
        Existing directory to save data and self to.
    D: int
        Dimensionality of the space.
    N: int
        Number of points to generate for the training set.
    N: int
        Number of points to generate for the test set.
    uninformative : boolean, optional [False]
        If True, then all class Gaussians have the same zero mean, making the
        cheap dimension features completely uninformative.
    """
    def __init__(self, dirname, D=2, N=1000, N_test=500, uninformative=False):
        self.dirname = dirname
        self.uninformative = uninformative

        Q = 2 ** D  # num quadrants
        cheap = 1
        expensive = 10
        self.actions = (
            ['d{}'.format(d) for d in range(D)] +
            ['q{}'.format(q) for q in range(Q)])
        self.action_costs = np.array(
            [cheap for d in range(D)] +
            [expensive for q in range(Q)])
        self.action_dims = (
            [1 for d in range(D)] +
            [Q for q in range(Q)])

        self.max_budget = cheap * D + expensive
        self.labels = range(Q)
        self.K = len(self.labels)
        self.D = D
        self.Q = Q
        self.N = N
        self.N_test = N_test
        self.validate()
        self.quadrant_indicators = np.array([x for x in itertools.product(*([(-1, 1)] * self.D))])

        # The data does not need to be standardized, because of its structure.
        X, y, coordinates = self.generate(N)
        X_test, y_test, _ = self.generate(N_test)

        self.data_filename = self.dirname + '/{}.h5'.format(self.budgetless_name)
        with h5py.File(self.data_filename, 'w') as f:
            f.create_dataset('X', data=X)
            f.create_dataset('y', data=y)
            f.create_dataset('coordinates', data=coordinates)
            f.create_dataset('X_test', data=X_test)
            f.create_dataset('y_test', data=y_test)

    def __config__(self):
        assert(isinstance(self, tc.data_sources.SyntheticOrthants))
        config = super(tc.data_sources.SyntheticOrthants, self).__config__()
        print config
        config['uninformative'] = self.uninformative
        return config

    @property
    def name(self):
        return 'synthetic_orthants_D{}_{}_N{}_Nt{}_{}'.format(
            self.D, self.uninformative, self.N, self.N_test, self.max_budget)

    def generate(self, N):
        """
        Generate X, y. Also return coordinates for plotting.
        """
        # Make sure the number of points is divisible by K for even distribution
        # of labels.
        N = N - np.mod(N, self.K)
        y = np.repeat(self.labels, N / self.K)
        X = np.zeros((N, self.D + self.Q ** 2))

        if self.uninformative:
            gaussian_means = np.zeros((N, self.D))
        else:
            gaussian_means = np.repeat(self.quadrant_indicators, N / self.K, axis=0)
        gaussian_std = 1
        coordinates = np.random.randn(N, self.D) * gaussian_std + gaussian_means

        X[:, :self.D] = np.sign(coordinates)
        X[:, self.D:] = np.random.randn(N, self.Q ** 2) / 2
        for q in range(self.Q):
            ind = (X[:, :self.D] * self.quadrant_indicators[q, :]).sum(1) > 0
            s = X[:, slice(*self.feature_bounds['q{}'.format(q)])]
            s[ind, :] = 0
            s[ind, y[ind]] = 1
        return X, y, coordinates

    def stats(self):
        """
        Output some statistics about the data distribution.
        """
        for i in range(self.Q):
            num_with_label = (self.y == self.labels[i]).sum()
            num_with_label_in_quadrant = np.all(self.X[self.y == self.labels[i], :self.D] == self.quadrant_indicators[i], axis=1).sum()
            print('Label {}'.format(self.labels[i]))
            print('  fraction of total with label: {:.3f}'.format(1. * num_with_label / self.X.shape[0]))
            print('  fraction in correct quadrant: {:.3f}'.format(1. * num_with_label_in_quadrant / num_with_label))

    def plot_source_data(self):
        """
        Plot the distribution of points in the space, colored by label.
        """
        D = self.D
        coord = self.coordinates
        y = self.y
        labels = self.labels

        if D > 3:
            print("No way to plot data with more than three dimensions.")
            return
        fig = plt.figure(figsize=(5, 5))
        if D == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        markers = ['o', 'v', 'D', 's', '.', '<', '>', 'x']
        for i in range(len(labels)):
            if D == 3:
                ax.plot(coord[y == i, 0], coord[y == i, 1], coord[y == i, 2],
                        'o', alpha=0.35, label=labels[i], marker=markers[i])
            elif D == 2:
                ax.plot(coord[y == i, 0], coord[y == i, 1],
                        'o', alpha=0.2, label=labels[i], marker=markers[i])
            else:
                ax.plot(coord[y == i], np.zeros_like(coord[y == i]),
                        'o', alpha=0.1, label=labels[i])
        plt.xlabel('$d_0$')
        ax.set_xlim([-D, D])
        plt.ylabel('$d_1$')
        ax.set_ylim([-D, D])
        if D == 3:
            ax.set_zlabel('$d_3$')
            ax.set_zlim([-D, D])
        else:
            plt.legend(fancybox=True)
        ax.patch.set_facecolor('none')
        return fig

    def plot_data_matrix(self):
        """
        Plot a random sample of the data matrix.
        """
        D = self.D
        if D > 4:
            print("Feature matrix is too high-dimensional to display.")
            return

        X = self.X
        N = X.shape[0]
        fig = plt.figure()
        plt.matshow(X[np.random.choice(range(N), size=50), :], cmap=plt.cm.RdBu_r)
        plt.colorbar()
        return fig

if __name__ == '__main__':
    so = SyntheticOrthants('data/data_sources', D=4)
    so.stats()
    so.save()
    #fig = so.plot_source_data()
    #fig = so.plot_data_matrix()
    #plt.show()
