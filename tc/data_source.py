import types
import numpy as np
from collections import OrderedDict
import json
import cPickle as pickle
import h5py


class DataSource(object):
    """
    Data is a class that defines actions and provides their observations
    for a set of data.

    Notation
    --------
    - A is the number of actions
    - D_a is the dimensionality of the observations of action a
    - D is the sum of all D_a
    - K is the number of classes
    - C is the cost budget
    - T is the total cost of all actions

    Properties
    ----------
    actions: string list of length A
        Names of the possible actions.

    action_dims: int sequence of length A
        Feature dimensions of each possible action.

    action_costs: (A,) ndarray of float
        Average costs to compute each feature, in seconds.

    labels: sequence of length K
        The possible class labels.

    max_budget: int
        If -1, no maximum budget.
    """
    def validate(self):
        assert(hasattr(self, 'actions'))
        assert(isinstance(self.actions, types.ListType))

        assert(hasattr(self, 'action_dims'))
        for d in self.action_dims:
            assert(isinstance(d, types.IntType))

        assert(hasattr(self, 'action_costs'))
        assert(isinstance(self.action_costs, np.ndarray))
        for c in self.action_costs:
            assert(isinstance(c, types.IntType) or
                   isinstance(c, types.FloatType))

        assert(isinstance(self.max_budget, types.IntType) or
               isinstance(self.max_budget, types.FloatType))

    def save(self):
        """
        Save self to canonically named pickle file in self.dirname.

        Returns
        -------
        filename: string
        """
        filename = self.dirname + '/{}.pickle'.format(self.name)
        with open(filename, 'w') as f:
            pickle.dump(self, f, protocol=2)
        return filename

    def save_for_greedy_miser(self):
        """
        Save in .mat format for use with Greedy Miser code.
        """
        pass

    @property
    def feature_bounds(self):
        """
        Return bounds for action_dims for lookup of instance data.
        If these have not been computed yet, computes and caches them.

        Use like this:
        tc.util.slice_array(ds.X[0], ds.feature_bounds, ds.actions[0])
        """
        if not hasattr(self, 'feature_bounds_'):
            bounds = np.hstack((0, np.cumsum(self.action_dims, dtype='int')))
            self.feature_bounds_ = dict(zip(self.actions, zip(bounds[:-1], bounds[1:])))
        return self.feature_bounds_

    @property
    def budgetless_name(self):
        return '_'.join(self.name.split('_')[:-1])

    def __getattr__(self, name):
        """
        If the following properties are called, fetch from data file.
        Otherwise, maintain default behavior of raising AttributeError.
        """
        if name in ['X', 'y', 'X_test', 'y_test', 'coordinates']:
            with h5py.File(self.data_filename, 'r') as f:
                data = f.get(name)[:]
            return data
        else:
            raise AttributeError

    def __config__(self):
        return OrderedDict([
            ('num_labels', len(self.labels)),
            ('num_actions', len(self.actions)),
            ('total_dims', sum(self.action_dims)),
            ('total_cost', sum(self.action_costs)),
            ('max_budget', self.max_budget),
            ('actions', str(self.actions)),
            ('action_dims', str(self.action_dims)),
            ('action_costs', str(self.action_costs)),
            ('labels', str(self.labels)),
            ('N', self.N),
            ('N_test', self.N_test)
        ])

    def __repr__(self):
        return json.dumps(self.__config__(), indent=4)
