"""tax

In this module we provide the bird taxonomy (courtesy of Dr Ryan Farrell) for
the CUB-200 classes, the taxonomy for the Cifar-100 dataset (corresponding to
the python format), as well as the code to generate the pairwise information
gain between the classes, assuming each class to have the same size.
"""
import cPickle as pickle
import json
import networkx as nx
import numpy as np
from os import path
import wordnet

__BIRD_FILENAME = path.join(path.dirname(__file__), 'taxCUB.json')
__ROOT_NAME = 'Aves'
__CIFAR_FILENAME = path.join(path.dirname(__file__),
        'cifar_100_tax.pickle')
__IMAGENET_FILENAME = path.join(path.dirname(__file__),
        'imagenet_%d_tax.pickle')

def _add_bird_node(graph, node):
    """This is an internal function that is called by get_bird_taxonomy to
    add all the subnodes under a given node to the graph
    """
    if node['level'] == 'genus':
        # simply add all the species under this genus
        for child in node['children']:
            # the id_tax contains a slash which we remove here.
            graph.add_edge(node['sci_name'], child['id_tax'][:-1])
    else:
        for child in node['children']:
            graph.add_edge(node['sci_name'], child['sci_name'])
            _add_bird_node(graph, child)
    return

def get_bird_taxonomy():
    """Obtains the bird taxonomy from the json file provided in birdtax
    """
    data = json.load(open(__BIRD_FILENAME, 'r'))['JSON Tree']
    graph = nx.DiGraph()
    _add_bird_node(graph, data)
    return graph

def get_cifar_taxonomy():
    graph = pickle.load(open(__CIFAR_FILENAME))
    return graph

def get_imagenet_taxonomy(num):
    """The num should either be 7404 or 1000
    """
    graph = pickle.load(open(__IMAGENET_FILENAME % num))
    return graph

def _compute_imagenet_taxonomy(num):
    """This function is used to compute the imagenet 7404 taxonomy from
    imagenet_meta.txt, and won't be needed after we get the imagenet tax
    pickle file.
    """
    filename = path.join(path.dirname(__file__), 'imagenet_meta%d.txt' % num)
    lines = [line.strip() for line in open(filename)]
    graph = nx.DiGraph()
    for i in range(0, len(lines), 4):
        wnid = lines[i]
        graph.add_node(wnid)
        graph.node[wnid]['word'] = lines[i+1]
        graph.node[wnid]['gross'] = lines[i+2]
        children = lines[i+3].split(' ')
        for child in children[1:]:
            graph.add_edge(wnid, child)
    return graph

def get_bird_names(graph = None):
    """Get the list of bird class names
    """
    if graph is None:
        graph = get_bird_taxonomy()
    classes = [n for n in graph.nodes() if len(graph.successors(n)) == 0]
    classes.sort()
    return classes

def get_leaves_sorted(graph):
    """Get the list of leaves so that when following this order, any subtree
    in the taxonomy will be a contiguous segment in the list.

    You need to make sure the graph passed in is a tree. For example, do NOT
    run this on imagenet!
    """
    root = [n for n in graph.nodes() if len(graph.predecessors(n)) == 0]
    root = root[0]
    return _get_leaves_sorted(graph, root)

def _get_leaves_sorted(graph, root):
    """ The recursive routine for get_bird_names_tree_sorted.
    """
    children = graph.successors(root)
    if len(children) == 0:
        return [root]
    else:
        leaves = []
        leaves_uniq = set()
        for child in children:
            for leaf in _get_leaves_sorted(graph, child):
                if leaf in leaves_uniq:
                    pass
                else:
                    leaves.append(leaf)
                    leaves_uniq.add(leaf)
        return leaves

def get_subgraph(graph, node):
    """Get the subgraph rooted at node. Note that if there are loops in the
    graph, this function might not work.
    """
    queue = [node]
    idx = 0
    visited = set()
    nodes = set(queue)
    while idx < len(queue):
        # expand the current node in queue
        children = graph.successors(queue[idx])
        visited.add(queue[idx])
        nodes.update(children)
        queue.extend(c for c in children if c not in visited)
        idx += 1
    return graph.subgraph(nodes)

def visualize_subgraph(graph, node, prog = "twopi", **kwargs):
    """Visualizes the subgraph rooted at node.
    """
    subgraph = get_subgraph(graph, node)
    nx.draw_graphviz(subgraph, prog, **kwargs)

def draw_bird_taxonomy(filename, graph = None):
    """Draws the bird taxonomy. If graph is None, the code loads the data from
    the json file.
    """
    import pygraphviz as pgv
    if graph is None:
        graph = get_bird_taxonomy()
    agraph = nx.to_agraph(graph)
    agraph.graph_attr['rankdir'] = 'LR'
    agraph.layout('dot')
    agraph.draw(filename)

def compute_leave_sets(graph, key = 'leaves'):
    """For each node, compute the set of leaves under it, and store it under
    the given key ('leaves' in default).

    We made this function non-recursive for large graphs.
    """
    if len(graph.nodes()) == 0:
        return
    stack = [n for n in graph.nodes() if len(graph.predecessors(n)) == 0]
    expanded = set()
    if len(stack) == 0:
        raise ValueError, "The graph does not have any root: possible cycles?"
    while len(stack) > 0:
        if stack[-1] in expanded:
            # all children in this stack has been computed, so we can compute
            # the node now.
            if len(graph.successors(stack[-1])) == 0:
                # I am a leaf
                graph.node[stack[-1]][key] = set([stack[-1]])
            else:
                leaves = set()
                for c in graph.successors(stack[-1]):
                    leaves.update(graph.node[c][key])
                graph.node[stack[-1]][key] = leaves
            del stack[-1]
        else:
            expanded.add(stack[-1])
            stack += graph.successors(stack[-1])
    return

def get_subgraph_with_leaves(graph, leaves):
    """get the subgraph with the given nodes as leaves. All ancestors of the
    leaves will be preserved, and all descendents of the leaves will be
    discarded.
    """
    nodes = list(leaves)
    visited = set(leaves)
    i = 0
    while i < len(nodes):
        predecessors = graph.predecessors(nodes[i])
        for p in predecessors:
            if p not in visited:
                nodes.append(p)
                visited.add(p)
        i += 1
    return graph.subgraph(nodes)


def pairwise_info_gain(graph, root = None):
    """Compute the info gain between the leaf nodes for a given graph. The
    function returns a dictionary where the pairwise info gain between leaf
    nodes can be accessed using a tuple.

    During the execution of the graph, we create a key "_leaves" for each
    node in the graph, which contains the list of leaf nodes under the node.
    It is cleaned once we have the information gain computed.
    """
    infogain = {}
    if root is None:
        root = [n for n in graph.nodes() if len(graph.predecessors(n)) == 0]
        if len(root) != 1:
            raise ValueError, "The graph must only have one root!"
        root = root[0]
    _pairwise_info_gain(graph, root, infogain)
    # in the end, we need to compute the offset, which is the log of the size
    # of the tree
    offset = np.log(len(graph.node[root]['_leaves']))
    del graph.node[root]['_leaves']
    for key in infogain:
        infogain[key] = offset - infogain[key]
    # for each key (a,b), we add key (b,a) as well
    for key in infogain.keys():
        infogain[(key[1],key[0])] = infogain[key]
    return infogain

def _pairwise_info_gain(graph, node, infogain):
    """This is the main routine that carries out the pairwise info gain
    computation. It also works on DAGs - when the hierarchy is not a tree.
    """
    if len(graph.successors(node)) == 0:
        graph.node[node]['_leaves'] = set([node])
        infogain[(node,node)] = 0.
        return
    children = graph.successors(node)
    # run pairwise info gain for the children first
    for c in children:
        _pairwise_info_gain(graph, c, infogain)
    leaves = set()
    for c in children:
        leaves.update(graph.node[c]['_leaves'])
    graph.node[node]['_leaves'] = leaves
    size = len(graph.node[node]['_leaves'])
    gain = np.log(size)
    for i in range(len(children)):
        for j in range(i+1, len(children)):
            leaves_i = graph.node[children[i]]['_leaves']
            leaves_j = graph.node[children[j]]['_leaves']
            for leaf_1 in leaves_i:
                for leaf_2 in leaves_j:
                    if (leaf_1,leaf_2) not in infogain and \
                            (leaf_2, leaf_1) not in infogain:
                        infogain[(leaf_1, leaf_2)] = gain
    # after computing the pairwise infogain between children, the _leaves
    # key is no longer needed
    for c in children:
        del graph.node[c]['_leaves']
    return

def bird_info_gain():
    """This function returns a 200*200 matrix that contains the information
    gain between each bird class.
    """
    mat = np.zeros((200,200))
    infogain = pairwise_info_gain(get_bird_taxonomy(), root = __ROOT_NAME)
    for key in infogain:
        i = int(key[0][:3]) - 1
        j = int(key[1][:3]) - 1
        mat[i,j] = infogain[key]
    return mat

def cifar_info_gain():
    """This function returns a 100*100 matrix that contains the information gain
    between each cifar class.
    """
    mat = np.zeros((100,100))
    infogain = pairwise_info_gain(get_cifar_taxonomy())
    for key in infogain:
        mat[key[0],key[1]] = infogain[key]
    return mat

def get_bird_ancestor_matrix(graph = None):
    """Returns a matrix of [num_nodes * num_leaves], where the leaves are 
    ordered from 1 to 200 (0 to 199, in c index).
    """
    if graph is None:
        graph = get_bird_taxonomy()
    node_id_map = dict((n,i) for i,n in enumerate(graph.nodes()))
    leaves = [n for n in graph.nodes() if len(graph.successors(n)) == 0]
    mat = np.zeros((len(graph.nodes()), len(leaves)))
    for leaf in leaves:
        leafid = int(leaf[:3]) - 1
        current = leaf
        while current != __ROOT_NAME:
            mat[node_id_map[current], leafid] = 1
            current = graph.predecessors(current)[0]
        mat[node_id_map[__ROOT_NAME], leafid] = 1
    return mat

def get_cifar_ancestor_matrix():
    """This funtion returns a 120*100 matrix that contains the ancestor
    relationship of Cifar.
    """
    graph = get_cifar_taxonomy()
    mat = np.zeros((120, 100))
    for u,v in graph.edges():
        if v < 100:
            mat[v,v] = 1.
            mat[u,v] = 1.
    mat[-1,:] = 1.
    return mat

def compute_erlang_prior(graph, lamda, normalize = True):
    """Compute the erlang prior for each node in the tree. The prior will be
    based on the "size": the number of leaf nodes in the subtree rooted at
    each node.
    Output:
        prior: a dictionary where keys are the node names and the values are
            the probability values. If normalize is False, we do not do prior
            normalization.
    """
    prior = {}
    compute_leave_sets(graph)
    for n in graph.nodes():
        s = float(len(graph.node[n]['leaves'])) / lamda
        prior[n] = s / lamda * np.exp(-s)
    if normalize:
        total = sum(prior.values())
        for n in prior:
            prior[n] /= total
    return prior

def get_topological_order(graph):
    """Get the topological order for a specific graph
    """
    order = []
    idx = 0
    predecessor_count = {}
    # initialize predecessor count
    for n in graph.nodes():
        count = len(graph.predecessors(n))
        if count == 0:
            order.append(n)
        else:
            predecessor_count[n] = count
    # process the list
    while (idx < len(order)):
        for n in graph.successors(order[idx]):
            if predecessor_count[n] == 1:
                order.append(n)
                del predecessor_count[n]
            else:
                predecessor_count[n] -= 1
        idx += 1
    # finally, check if the graph is a DAG: if not, we will have
    # remaining nodes unable to be inserted to the order
    if len(order) != len(graph.nodes()):
        raise ValueError, "The graph does not seem to be a DAG."
    return order
            

if __name__ == '__main__':
    pass
