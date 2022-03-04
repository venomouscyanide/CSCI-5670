"""
Author: Paul Louis, <paul.louis@ontariotechu.ca>
An attempt at summaring and visualizing UOIT information network
"""
import math

import networkx as nx
from networkx import condensation
import matplotlib.pyplot as plt


def draw_condensed(graph):
    scc = list(nx.strongly_connected_components(graph))
    condensed = condensation(graph, scc)

    scc_map = dict()
    for index, component in enumerate(scc):
        node_label = list(sorted(component, key=lambda x: graph.degree(x), reverse=True))[0]
        scc_map[index] = node_label

    nx.set_node_attributes(condensed, {k: v for k, v in condensed.degree}, name="degree")
    summary_graph = nx.snap_aggregation(condensed, node_attributes=['degree'])

    updated_mapping = {}
    for node in summary_graph.nodes():
        node_attr = summary_graph.nodes[node]
        groups = list(node_attr['group'])
        highest_degree_label_val = sorted(groups, key=lambda x: graph.degree(scc_map[x]), reverse=True)[0]
        highest_degree_label = scc_map[highest_degree_label_val]
        updated_mapping[node] = highest_degree_label[:42]

    summary_graph = nx.relabel_nodes(summary_graph, updated_mapping)

    pos = nx.spring_layout(summary_graph,
                           k=5 / math.sqrt(summary_graph.order()))  # https://stackoverflow.com/a/34632757
    f = plt.figure(1, figsize=(16, 16))
    nx.draw(summary_graph, pos=pos, with_labels=True)
    plt.show()
    f.savefig("vizinfo.pdf", bbox_inches='tight')

    # failed attempt at saving as gexf to be processed by Gephi
    # for node in summary_graph.nodes:
    #     summary_graph.nodes[node].pop('group')
    #     # summary_graph.nodes[node]['group'] = list(summary_graph.nodes[node]['group'])
    #
    print(summary_graph.number_of_nodes(), summary_graph.number_of_edges())
    # nx.write_gexf(summary_graph, "vizinfo.gexf")


if __name__ == '__main__':
    graph = nx.read_gexf('interim_20000_len_graph.gexf')
    draw_condensed(graph)
