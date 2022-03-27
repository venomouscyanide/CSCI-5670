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
    print(f"Pagerank Converged. No of iterations run: {k}")
    return k


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
    # zeros -> 6, 11, 12, 17

    # answer to question 4.b
    pr_k_2 = page_rank_analysis(ajc_matrix, scaled=False, draw=False, k=2)
    pr_k_3 = page_rank_analysis(ajc_matrix, scaled=False, draw=False, k=3)

    # answer to question 4.c
    s = 0.9  # scaling factor
    pr_k_conv = page_rank_analysis(ajc_matrix, scaled=False, draw=False, k=math.inf, s=s)

    # answer to question 4.f
    s = 0.9  # scaling factor
    scaled_pr_k = page_rank_analysis(ajc_matrix, scaled=True, draw=False, k=math.inf, s=s)
