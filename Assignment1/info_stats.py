import itertools
import json
import operator
from collections import defaultdict

import networkx as nx
import pandas as pd
from networkx import density, average_clustering, average_shortest_path_length, diameter, degree_centrality, \
    closeness_centrality, betweenness_centrality, pagerank

import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities, girvan_newman
import networkx.algorithms.community as nx_comm


def get_degree(graph):
    return graph.number_of_nodes()


def get_edges(graph):
    return graph.number_of_edges()


def plot_deg_dist(log_log, graph):
    freq_count = defaultdict(int)
    for url, degree in graph.degree():
        freq_count[degree] += 1

    plt.figure(1)
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution')

    plt.scatter(x=freq_count.keys(), y=freq_count.values(), marker='.')
    if log_log:
        plt.yscale('log')
        plt.xlabel('Log Degree')
        plt.ylim(1, max(freq_count.values()))
        plt.title('Log-Log Degree Distribution')
        plt.xscale('log')
        plt.ylabel('Log Count')
        plt.xlim(1, max(freq_count.keys()))
    plt.show()


def connected_component_analysis(graph):
    largest_scc = max(nx.strongly_connected_components(graph), key=len)
    largest_wcc = max(nx.weakly_connected_components(graph), key=len)

    num_nodes_scc = len(largest_scc)
    scc_sub_graph = graph.subgraph(largest_scc)
    avg_path_len_scc = average_shortest_path_length(scc_sub_graph)
    diameter_scc = diameter(scc_sub_graph)

    print("SCC Stats")
    print(f"No nodes: {num_nodes_scc}")
    print(f"avg_path_len_scc: {avg_path_len_scc}")
    print(f"diameter_scc: {diameter_scc}", end='\n \n')

    num_nodes_wcc = len(largest_wcc)
    wcc_sub_graph = graph.subgraph(largest_wcc).to_undirected()
    avg_path_len_wcc = average_shortest_path_length(wcc_sub_graph)
    diameter_wcc = diameter(wcc_sub_graph)

    print("WCC Stats")
    print(f"No nodes: {num_nodes_wcc}")
    print(f"avg_path_len_scc: {avg_path_len_wcc}")
    print(f"diameter_scc: {diameter_wcc}", end='\n \n')


def community_analysis(graph):
    # add breakpoints and analyze the communities in-memory
    # greedy modularity based

    greedy_communities = greedy_modularity_communities(graph, n_communities=15)
    comm_dict = {}
    for index, comm in enumerate(greedy_communities):
        comm_dict.update(
            {index: {k: graph.degree(k) for k in list(sorted(comm, key=lambda x: graph.degree(x), reverse=True))[:3]}}
        )

    with open('community_detection.json', 'w') as file:
        file.write(json.dumps(comm_dict))
    # comp = girvan_newman(graph)
    # limited = itertools.takewhile(lambda c: len(c) <= 15, comp)
    # for communities in limited:
    #     print(tuple(sorted(c) for c in communities))
    # lovain_communities = nx_comm.louvain_communities(graph) # this is broken


def _centrality_helper(type, cent_dict, df, top):
    print(f"Running {type} for top={top}")
    top_bottom_10 = list(sorted(cent_dict.items(), key=operator.itemgetter(1), reverse=True))
    if top:
        top_bottom_10 = top_bottom_10[:10]
    else:
        top_bottom_10 = top_bottom_10[::-1][:10]
    start = 1 if top else len(cent_dict) - 9
    for index, (label_value) in enumerate(top_bottom_10, start=start):
        df.loc[len(df)] = [index, type, round(label_value[1], 5), label_value[0]]


def get_top_least_centrality_nodes(graph, name, top=True):
    df = pd.DataFrame(
        columns=["Rank", "Type", "Centrality Value", "Node Label"])

    dc = degree_centrality(graph)
    _centrality_helper("Degree Centrality", dc, df, top)

    cc = closeness_centrality(graph)
    _centrality_helper("Closeness Centrality", cc, df, top)

    bc = betweenness_centrality(graph)
    _centrality_helper("Betweeness Centrality", bc, df, top)

    pr = pagerank(graph)
    _centrality_helper("Page Rank", pr, df, top)

    df.to_csv(name, index=False)


def get_stats(graph):
    num_nodes = get_degree(graph)  # no of nodes in the graph
    print(f"Number of nodes: {num_nodes}", end='\n \n')
    num_edges = get_edges(graph)  # no of edges in the graph
    print(f"Number of edges: {num_edges}", end='\n \n')

    edge_density = density(graph)  # graph density
    print(f"Edge density: {edge_density}", end='\n \n')

    avg_cc = average_clustering(graph)
    print(f"Average Clustering Coefficient: {avg_cc}", end='\n \n')

    plot_deg_dist(False, graph)
    plot_deg_dist(True, graph)

    connected_component_analysis(graph)
    community_analysis(graph)

    get_top_least_centrality_nodes(graph, name="df_top.csv", top=True)
    get_top_least_centrality_nodes(graph, name="df_bottom.csv", top=False)


if __name__ == '__main__':
    # graph = nx.read_gexf('final_100000_len_graph.gexf')
    graph = nx.read_gexf('interim_20000_len_graph.gexf')  # smaller graph
    get_stats(graph)
