import numpy as np
import tc


class Random(tc.DataSource):
    """
    Generates random data, for quick testing.
    Random costs.

    Parameters
    ----------
    actions: list of string
    action_dims: sequence of int
    labels: list of string
    N: int
        Number of instances to generate.
    """
    def __init__(self, actions, action_dims, labels, N):
        self.actions = actions
        self.action_dims = action_dims
        self.action_costs = np.random.rand(len(self.actions))
        self.labels = labels
        self.X = np.random.rand(N, action_dims.sum())
        self.y = np.random.randint(len(self.labels), size=N)
        self.validate()
