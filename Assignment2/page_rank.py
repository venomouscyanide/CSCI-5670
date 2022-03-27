"""
Description : Helps compute scaled and unscaled PageRank matrices using the Easley and Kleinberg PageRank formulations
Course : CSCI 5760
Author : Paul Louis <paul.louis@ontariotechu.net>
Usage : python3.9 page_rank.py
"""
import math
import string  # optional
from typing import Union

import networkx as nx  # used for creation and interaction with graphs
import numpy as np  # used for matrix arithmetic
import matplotlib.pyplot as plt  # used for plotting the original graph
import pandas as pd  # used for creating CSV files
from networkx import relabel_nodes  # used to relabel networkx graph nodes
from numpy.linalg import linalg  # used for matrix power arithmetic


def _get_n_matrices(ajc_matrix, n, s):
    n_matrix = ajc_matrix.copy()  # Init N matrix as a copy of the original adjacency matrix
    n_matrix = n_matrix.astype(float)  # make sure the matrix is of float type for PR formulation

    n_prime_matrix = ajc_matrix.copy()  # Init N prime matrix as a copy of the original adjacency matrix
    n_prime_matrix = n_prime_matrix.astype(float)  # make sure the matrix is of float type for PR formulation

    for ind, row in enumerate(ajc_matrix):
        # this for-loop helps update the N and N prime matrix values to reflect the actual expected values

        non_zero = 0
        for value in row:  # simple way to get number of non-zero entries
            if value:
                non_zero += 1

        if non_zero:
            # if there are non zero rows in the adjacency matrix, we update N matrix row by creating fractions
            # the fractions created are just the values over total number of non-zero entries
            updated_row = row / non_zero
            n_matrix[ind] = updated_row

        else:
            # if there are no non zero entries, we create a self loop for the node
            updated_row = np.array(
                [1 if index == ind else 0 for index in range(n)]  # put 1 for the node who's row is being created
            )
            n_matrix[ind] = updated_row

        # create the scaled PR matrix row(multiply by S and add (1-S)/N)
        scaled_updated_row = updated_row * s + (1 - s) / n
        n_prime_matrix[ind] = scaled_updated_row
    return n_matrix, n_prime_matrix


def pr_until_conv(page_rank_matrix, n_matrix, verbose=False):
    prev = page_rank_matrix.copy()  # create a copy of PageRank matrix to act as initial previous values
    curr = np.array([0 for _ in range(
        n_matrix.shape[0])])  # current values are initialed to zeroes as it has not been calculated yet
    k = 1  # init number of iterations to 1

    while not np.allclose(curr, prev):
        # PageRank is said to have converged all previous values are within a tolerance value of 1.e-5 to current values
        if verbose:
            print(f"Running iteration: {k}")

        pr_k_iter = np.matmul(linalg.matrix_power(n_matrix.T, k), page_rank_matrix)  # normal page range formulation

        # prev value of PR is just formulated with k - 1
        prev = np.matmul(linalg.matrix_power(n_matrix.T, k - 1), page_rank_matrix)
        curr = pr_k_iter.copy()  # current value of PR is a copy of what was just calculated

        k += 1  # increase number of iterations

    # number of iterations will be always one above due to exit condition above. So, subtract 1 to get actual value
    k -= 1
    if verbose:
        print(f"Pagerank Converged. No of iterations run: {k}")
    return curr


def draw_graph(graph):
    # helps draw a graph object and save it as a png file
    f = plt.figure(1, figsize=(16, 16))
    nx.draw(graph, with_labels=True, pos=nx.spring_layout(graph))
    plt.show()  # check if same as in the doc visually
    f.savefig("input_graph.pdf", bbox_inches='tight')


def page_rank_analysis(ajc_matrix, scaled=False, draw=False, k: Union[float, int] = 1, s=0.0):
    graph = nx.from_numpy_matrix(ajc_matrix, create_using=nx.DiGraph)  # create a nx graph object
    # t = list(string.ascii_lowercase) # helps in viz textbook eg(optional)
    # graph = relabel_nodes(graph, mapping={k: t[k].upper() for k in range(18)}) # optional
    graph = relabel_nodes(graph, mapping={k: k + 1 for k in range(18)})  # relabel the graph nodes to start from label 1

    if draw:
        # draw the graph to compare with the graph in the assignment doc(sanity check)
        draw_graph(graph)

    n = graph.number_of_nodes()  # get number of nodes in the graph
    n_matrix, n_prime_matrix = _get_n_matrices(ajc_matrix, n, s)  # calculate N and N prime matrices for PageRank

    # create initial PR matrix. Each value is 1/number of nodes. In our question it's 1/18
    page_rank_matrix = np.array([1 / n for _ in range(n)])
    page_rank_matrix.resize([n, 1])

    if not math.isinf(k):
        # do not run PR until convergence. Run for k iterations
        if scaled:
            # the scaled pagerank formulation is just (NPrimeMatrix.Transpose)^k.InitialPRMatrix
            pr_k_iter = np.matmul(linalg.matrix_power(n_prime_matrix.T, k), page_rank_matrix)
        else:
            # the pagerank formulation is just (NMatrix.Transpose)^k.InitialPRMatrix
            pr_k_iter = np.matmul(linalg.matrix_power(n_matrix.T, k), page_rank_matrix)
    else:
        # Run PR until convergence
        if scaled:
            # scaled pagerank formulation until convergence
            pr_k_iter = pr_until_conv(page_rank_matrix, n_prime_matrix)
        else:
            # pagerank formulation until convergence
            pr_k_iter = pr_until_conv(page_rank_matrix, n_matrix)

    return pr_k_iter


def perturb_ajc_analysis(ajc_matrix):
    # delete one edge at a time and observe the creation of sinks

    print("-----------")
    table_for_report = pd.DataFrame(columns=["New Sink", "Edge Deleted for Sink Creation"])  # used for creating table

    original_pr_conv = page_rank_analysis(ajc_matrix, scaled=False, draw=False, k=math.inf)  # original converged PR
    expected_sinks = np.where(np.around(original_pr_conv, decimals=5) != 0)[0]  # default sinks that occur
    org_non_zero = len(expected_sinks)  # original number of sinks expected

    for row_ind, row in enumerate(ajc_matrix):  # go over the rows
        for col_ind, row_col_value in enumerate(row):  # go over the columns in the row
            if row_col_value:  # if an edge exists, it should be deleted
                copy_adj = ajc_matrix.copy()  # copy the adjacency matrix for deleting the edge
                copy_adj[row_ind, col_ind] = 0  # edge deleted

                pr_k_conv = page_rank_analysis(copy_adj, scaled=False, draw=False, k=math.inf)  # new converged PR
                current_sinks = np.where(np.around(pr_k_conv, decimals=5) != 0)[0]  # new sinks after perturbation
                curr_non_zero_pr = len(current_sinks)  # number of new sinks
                if curr_non_zero_pr > org_non_zero:  # if new sinks introduced add to the table
                    print(f"Node {row_ind + 1} , edge:{row_ind + 1}->{col_ind + 1}")
                    print(f"New Sinks: {set(current_sinks + 1) - set(expected_sinks + 1)}")
                    table_for_report.loc[len(table_for_report)] = [row_ind + 1, f"({row_ind + 1},{col_ind + 1})"]
                    print("-----------")
    table_for_report.to_csv("sink_table.csv", index=False)  # save as CSV to be added to the report


if __name__ == '__main__':
    # eg from https://cs.brown.edu/courses/cs016/static/files/assignments/projects/GraphHelpSession.pdf slide 9
    # ajc_matrix = np.array([[0, 0, 0, 0],
    #                        [1, 0, 0, 0],
    #                        [1, 0, 0, 0],
    #                        [1, 0, 0, 0]])
    # Eg from textbook
    # ajc_matrix = np.array([[0, 1, 1, 0, 0, 0, 0, 0],
    #                       [0, 0, 0, 1, 1, 0, 0, 0],
    #                       [0, 0, 0, 0, 0, 1, 1, 0],
    #                       [1, 0, 0, 0, 0, 0, 0, 1],
    #                       [1, 0, 0, 0, 0, 0, 0, 1],
    #                       [1, 0, 0, 0, 0, 0, 0, 0],
    #                       [1, 0, 0, 0, 0, 0, 0, 0],
    #                       [1, 0, 0, 0, 0, 0, 0, 0]])
    # the adjacency matrix manually created for the assignment question
    ajc_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    # answer to question 4.a
    pr_k_1 = page_rank_analysis(ajc_matrix, scaled=False, draw=False, k=1)
    zero_pr = np.where(pr_k_1 == 0)[0]
    zero_pr = zero_pr + 1  # fix node label to match graph given

    # answer to question 4.b
    pr_k_2 = page_rank_analysis(ajc_matrix, scaled=False, draw=False, k=2)
    pr_k_3 = page_rank_analysis(ajc_matrix, scaled=False, draw=False, k=3)

    # answer to question 4.c
    pr_k_conv = page_rank_analysis(ajc_matrix, scaled=False, draw=False, k=math.inf)

    # answer to question 4.d
    perturb_ajc_analysis(ajc_matrix)

    # answer to question 4.e
    ajc_matrix_copy = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],  # add (2,11) (2,12) (2,17)
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # add (10,6)
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    pr_k_conv_new = page_rank_analysis(ajc_matrix_copy, scaled=False, draw=False, k=math.inf)

    # answer to question 4.f
    s = 0.9  # scaling factor
    scaled_pr_k = page_rank_analysis(ajc_matrix, scaled=True, draw=False, k=math.inf, s=s)

    print("Fin.")
