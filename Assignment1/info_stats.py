from collections import defaultdict

import networkx as nx
from networkx import density, average_clustering, average_shortest_path_length, diameter

import matplotlib.pyplot as plt


def get_degree(graph):
    return graph.number_of_nodes()


def get_edges(graph):
    return graph.number_of_edges()


def plot_deg_dist(log_log):
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
    if log_log:
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


def get_stats(graph):
    num_nodes = get_degree(graph)  # no of nodes in the graph
    print(f"Number of nodes: {num_nodes}", end='\n \n')
    num_edges = get_edges(graph)  # no of edges in the graph
    print(f"Number of edges: {num_edges}", end='\n \n')

    edge_density = density(graph)  # graph density
    print(f"Edge density: {edge_density}", end='\n \n')

    avg_cc = average_clustering(graph)
    print(f"Average Clustering Coefficient: {avg_cc}", end='\n \n')

    plot_deg_dist(log_log=False)
    plot_deg_dist(log_log=True)

    connected_component_analysis(graph)


if __name__ == '__main__':
    # graph = nx.read_gexf('final_100000_len_graph.gexf')
    graph = nx.read_gexf('interim_50000_len_graph_old.gexf')  # smaller graph
    get_stats(graph)
