import networkx as nx
from os import path

def get_wordnet_hierarchy(filename = None):
    """Get the DAG structure from the data.noun file from wordnet 3.0
    """
    graph = nx.DiGraph()
    if filename is None:
        filename = path.join(path.dirname(__file__), 'data.noun')
    fid = open(filename)
    for line in fid:
        if len(line) == 0 or line[0] == ' ':
            continue
        # split the line
        tokens = line.strip().split(' ')
        #token[0]: synset_offset
        #token[1]: lex_filenum
        #token[2]: ss_type
        #token[3]: w_cnt
        #token[4+int(token[3])*2]: p_cnt
        synset = 'n' + tokens[0]
        graph.add_node(synset)
        w_cnt = int(tokens[3], 16)
        graph.node[synset]['word'] = \
                ', '.join(tokens[4+i*2] for i in range(w_cnt))
        p_cnt_id = 4 + w_cnt * 2
        p_cnt = int(tokens[p_cnt_id])
        for i in range(p_cnt):
            symbol = tokens[p_cnt_id + 1 + i*4]
            other = 'n' + tokens[p_cnt_id + 2 + i*4]
            if symbol == '@':
                graph.add_edge(other, synset)
            elif symbol == '~':
                graph.add_edge(synset, other)
        # add gross
        graph.node[synset]['gross'] = \
                ' '.join(tokens[p_cnt_id + p_cnt * 4 + 2:])
    return graph



