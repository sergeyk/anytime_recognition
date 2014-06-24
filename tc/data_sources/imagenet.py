"""
Code to deal with the ImageNet dataset.

Feature details (computed by Yangqing):
    - VLfeat SIFT dense extraction: with SIFT patch size 16, 32 and 64,
        and a stride of 4 pixels. The images are reduced to size 500*500
        (smaller images are not resized up though).
    - LLC-coded features with 5-nearest neighbors, and a codebook of size 16k
    - Max-pooled on a 1x1 grid and 3x3 grid, so a total of 10 bins.
        This gives us 10 * 16k = 160k dimensional features.
    - Adagrad (multinomial) logistic regression on all training data.
"""
import os
import cPickle as pickle
import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import h5py
import tc
from tc import DataSource

# data/imagenet should contain classifier output data.
imagenet_dir = os.path.join(tc.repo_dir, 'data/imagenet')
imagenet_pickle_filename = os.path.join(os.path.dirname(__file__), 'imagenet_%d_tax.pickle')

# The Hedging Your Bets code release should be extracted to the ext/ folder.
# (Download from http://www.image-net.org/projects/hedging/).
ilsvrc65_meta_filename = tc.repo_dir + '/ext/hedging-1.0/code/ilsvrc65_meta.mat'

# I ran the MATLAB demo of the Hedging release and outut their classifier
# outputs on the validation and test sets of ILSVRC65.
ilsvrc65_clf_outputs_val = tc.repo_dir + '/test/support/ilsvrc65_clf_outputs_val.mat'
ilsvrc65_clf_outputs_test = tc.repo_dir + '/test/support/ilsvrc65_clf_outputs_test.mat'


def process_graph(g, nodes):
    """
    Parameters
    ----------
    g : networkx.DiGraph

    nodes : list
        List of node names, all of which are in g.
        The first K nodes in this list correspond to the leaf nodes, and are in
        the same order as observations that we may see.

    Returns
    -------
    graph : dict
        'g': the original graph, augmented with 'leaves' field
    """
    assert(len(g.nodes()) == len(nodes))
    N = len(nodes)

    # augment graph with 'leaves' field
    g = copy.deepcopy(g)
    compute_leave_sets(g, nodes)

    heights = np.array([g.node[node]['height'] for node in nodes]).flatten()
    leaf_mask = np.atleast_2d(heights == 0)
    K = leaf_mask.sum()

    # Construct K x N matrix of leaf membership.
    leaf_membership = np.zeros((K, N))
    leaf_membership[:, :K] = np.eye(K)
    for i in range(K, N):
        leaf_membership[g.node[nodes[i]]['leaf_inds'], i] = 1

    num_leaves = leaf_membership.sum(0)
    rewards = np.log2(K / num_leaves).flatten()

    graph = {
        'g': g, 'nodes': nodes, 'heights': heights, 'rewards': rewards,
        'leaf_membership': leaf_membership}
    return graph


def compute_leave_sets(graph, nodes):
    """
    For each node, compute the set of leaves under it, and store it under
    the given key.
    This function is able to handle large graphs, as it is not recursive.

    Code originally by my labmate Yangqing Jia.

    Parameters
    ----------
    graph : nx.DiGraph

    nodes : list of strings
    """
    leaves_key = 'leaves'
    leaf_inds_key = 'leaf_inds'

    if len(graph.nodes()) == 0:
        return
    stack = [n for n in graph.nodes() if len(graph.predecessors(n)) == 0]
    if len(stack) == 0:
        raise ValueError('The graph does not have any root: possible cycles?')
    expanded = set()
    while len(stack) > 0:
        node = stack[-1]
        if stack[-1] in expanded:
            # all children in this stack has been computed, so we can compute
            # the node now.
            if len(graph.successors(node)) == 0:
                # I am a leaf
                graph.node[node][leaves_key] = set([node])
            else:
                leaves = set()
                for c in graph.successors(node):
                    leaves.update(graph.node[c][leaves_key])
                graph.node[node][leaves_key] = leaves
            del stack[-1]
        else:
            expanded.add(node)
            stack += graph.successors(node)

    # also store indices to nodes list
    for node in nodes:
        inds = [nodes.index(n) for n in graph.node[node][leaves_key]]
        graph.node[node][leaf_inds_key] = sorted(inds)


class ILSVRC65(DataSource):
    """
    The ILSVRC65 dataset, as described in the Hedging Your Bets CVPR 2012 paper.
    """
    def __init__(self, dirname, max_budget=None):
        self.dirname = dirname
        self.graph = self.load_graph()
        g = self.graph['g']
        # TODO: is this the right way to determine names?
        self.labels = [g.node[node]['word'] for node in self.graph['nodes']]
        self.actions = [g.node[node]['word'] for node in np.array(self.graph['nodes'])[self.graph['heights'] == 0]]
        self.action_dims = np.ones(len(self.actions), dtype=int)
        self.action_costs = np.ones(len(self.actions), dtype=float)
        self.max_budget = max_budget
        if max_budget is None:
            self.max_budget = int(sum(self.action_costs) / 2)

        self.data_filename = self.dirname + '/ilsvrc65.h5'
        if not os.path.exists(self.data_filename):
            X, y, X_test, y_test = self.load_data()
            # NOTE: imagenet should not be standardized, because
            # the features are already all in [0,1] and the classifier
            # can be doing simple argmax over average of feature channels.
            # standardization ruins performance in that case.

            #ss = StandardScaler()
            #X = ss.fit_transform(X)
            #X_test = ss.transform(X_test)

            with h5py.File(self.data_filename, 'w') as f:
                f.create_dataset('X', data=X)
                f.create_dataset('y', data=y)
                f.create_dataset('X_test', data=X_test)
                f.create_dataset('y_test', data=y_test)
        self.N = self.X.shape[0]
        self.N_test = self.X_test.shape[0]

    @property
    def name(self):
        return 'ilsvrc65_{}'.format(self.max_budget)

    @staticmethod
    def load_graph():
        ilsvrc65 = loadmat(ilsvrc65_meta_filename)
        g65 = nx.DiGraph()
        # Load nodes in the order of the meta file (leaves first).
        nodes = []
        for x in ilsvrc65['synsets']:
            nodes.append(x['WNID'][0][0])
            g65.add_node(nodes[-1], {'word': x['words'][0][0], 'height': x['height'][0][0]})
        for x in ilsvrc65['synsets']:
            node = x['WNID'][0][0]
            for child in x['children'][0][0]:
                g65.add_edge(node, nodes[child - 1])
        graph = process_graph(g65, nodes)
        return graph

    @staticmethod
    def load_data():
        """
        Load the classifier outputs for val and test sets.
        """
        data = loadmat(ilsvrc65_clf_outputs_val)
        X = data['leaf_probs']
        y = data['labels'] - 1
        data = loadmat(ilsvrc65_clf_outputs_test)
        X_test = data['leaf_probs']
        y_test = data['labels'] - 1
        return X, y, X_test, y_test


class ImageNet(object):
    """
    DataSource for the ImageNet dataset.

    Parameters
    ----------
    max_leaves : int, optional
        The desired number of leaf nodes per group.
        Getting exactly this number in every group is extremely unlikely: instead, expect
        different smaller numbers close to this.

    method : string in ['random', 'size'], optional
        The method used to split the classes into groups.
        - 'random': leaf nodes are selected completely randomly.
        - 'size': all nodes are sorted by the number of leaf nodes they are
        ancestors to, and groups are formed by taking nodes smaller than
        max_leaves and removing the selected leaf nodes from all other nodes,
        iteratively.
    """
    def __init__(self, max_leaves=200, method='size'):
        num = 1000  # can be 7404 (for ImageNet10K?)
        self.g = pickle.load(open(imagenet_pickle_filename % num))
        self.nodes = self.g.nodes()
        if method == 'size':
            self.groups = self.select_groups_by_size(self.g, self.nodes, max_leaves)
        elif method == 'random':
            self.groups = self.select_groups_random
        else:
            raise('Unknown method!')
        self.labels = self.g.nodes()
        #self.X, self.y = self.load_data('val')

    @staticmethod
    def load_data(s):
        """
        Parameters
        ----------
        s : string in ['train', 'val', 'test']
        """
        assert(s in ['train', 'val', 'test'])
        X = np.load('data/imagenet/{0}_predict/{0}_prob.npy'.format(s))
        y = np.load('data/imagenet/{}_predict/labels_ascii_sorted.npy'.format(s))
        return X, y

    @staticmethod
    def select_groups_by_size(g, nodes, max_leaves):
        """
        Select leaf node groups by inner-node sizes.

        Parameters
        ----------
        g : networkx.DiGraph

        nodes : list of strings

        max_leaves : int
            The desired number of leaf nodes per group.
            Getting exactly this number in every group is extremely unlikely:
            instead, expect different smaller numbers close to this.

        Returns
        -------
        selected_groups : list of (node name, list of leaf node names) tuples.
        """
        g = copy.deepcopy(g)
        compute_leave_sets(g, nodes)
        selected_groups = []
        while True:
            all_num_leaves = [len(g.node[node]['leaves']) for node in g.nodes()]
            if sum(all_num_leaves) == 0:
                break
            ind = np.argsort(-np.array(all_num_leaves))
            sorted_nodes = np.array(g.nodes())[ind]
            for node in sorted_nodes:
                leaves = g.node[node]['leaves']
                if len(leaves) > 0 and len(leaves) <= max_leaves:
                    selected_groups.append((node, leaves))
                    for node_prime in np.array(g.nodes()):
                        if node_prime == node:
                            continue
                        g.node[node_prime]['leaves'] -= leaves
                    g.node[node]['leaves'] = set([])
                    break
        assert(sum(len(x[-1]) for x in selected_groups) == 1000)
        for group in selected_groups:
            node, leaves = group
            print(node, g.node[node]['word'], len(leaves))
        return selected_groups

    @staticmethod
    def select_groups_randomly(g, max_leaves):
        """
        Select leaf node groups randomly, such that all groups have max_leaves
        leaves (if not evenly divisible, the last group will be smaller).
        """
        # TODO implement
        pass

    def plot_groups(self, filename=None):
        return self.plot_selected_groups(self.g, self.groups, filename)

    @staticmethod
    def plot_selected_groups(g, selected_groups, filename=None):
        """
        Display the ImageNet DAG and highlight the selected groups
        in different colors.

        Parameters
        ----------
        g : networkx.DiGraph

        selected_groups : list of ('node', list of leaf nodes) tuples

        filename : string, optional
            If given, write plot out to filename.
        """
        fig = plt.figure(figsize=(12, 12))

        gray = (0.85, 0.85, 0.85, 1)
        graydark = (0.5, 0.5, 0.5, 1)
        ns = 30

        pos = nx.graphviz_layout(g, prog='sfdp')
        nx.draw_networkx_nodes(
            g, pos, with_labels=False, node_size=ns,
            node_color=gray, linewidth=None).set_edgecolor(graydark)
        root = [n for n in g.nodes() if g.predecessors(n) == []]
        nx.draw_networkx_nodes(
            g, pos, nodelist=root, with_labels=False,
            node_size=ns * 2, node_color=gray,
            linewidth=None).set_edgecolor(graydark)
        nx.draw_networkx_edges(g, pos, arrows=False, edge_color=[gray] * len(g.edges()))

        cmap = plt.get_cmap('Accent')
        colors = np.linspace(0, 1, len(selected_groups))
        darken = lambda c, r: (c[0] * r, c[1] * r, c[2] * r, c[3])
        for i, group in enumerate(selected_groups):
            nx.draw_networkx_nodes(g, pos, node_size=ns, nodelist=group[-1], node_color=cmap(colors[i])).set_edgecolor(darken(cmap(colors[i]), 0.75))

        plt.axis('equal')
        plt.axis('off')
        if filename is not None:
            plt.savefig(filename, dpi=300, facecolor='none')
        return fig

if __name__ == '__main__':
    ds = ImageNet()
    print(ds)
    fig = ds.plot_groups(filename="test.pdf")
