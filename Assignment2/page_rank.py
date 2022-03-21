import string

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx import relabel_nodes
from numpy.linalg import linalg


def _get_n_matrices(ajc_matrix, n):
    n_matrix = ajc_matrix.copy()
    n_matrix = n_matrix.astype(float)

    n_prime_matrix = ajc_matrix.copy()
    n_prime_matrix = n_prime_matrix.astype(float)

    for ind, row in enumerate(ajc_matrix):
        non_zero = 0
        for value in row:
            if value:
                non_zero += 1
        if non_zero:
            updated_row = row / non_zero
            n_matrix[ind] = updated_row

        else:
            updated_row = np.array([1 if index == ind else 0 for index in range(n)])
            n_matrix[ind] = updated_row

        scaled_updated_row = updated_row * s + (1 - s) / n
        n_prime_matrix[ind] = scaled_updated_row
    return n_matrix, n_prime_matrix


def pr_until_conv(page_rank_matrix, n_matrix, verbose=False):
    prev = page_rank_matrix.copy()
    curr = np.array([0 for _ in range(18)])
    k = 1
    while not np.allclose(curr, prev):
        if verbose:
            print(f"Running iteration: {k}")

        pr_k_iter = np.matmul(linalg.matrix_power(n_matrix.T, k), page_rank_matrix)

        prev = np.matmul(linalg.matrix_power(n_matrix.T, k - 1), page_rank_matrix)
        curr = pr_k_iter.copy()

        # print(np.sum(prev))
        # print(np.sum(curr))
        k += 1

    print(f"Done pagerank. No of iterations:{k - 1}")
    return curr


def page_rank_analysis(ajc_matrix, s, draw=False):
    graph = nx.from_numpy_matrix(ajc_matrix, create_using=nx.DiGraph)
    # t = list(string.ascii_lowercase) # helps in viz textbook eg
    # graph = relabel_nodes(graph, mapping={k: t[k].upper() for k in range(18)})
    graph = relabel_nodes(graph, mapping={k: k + 1 for k in range(18)})

    if draw:
        f = plt.figure(1, figsize=(16, 16))
        nx.draw(graph, with_labels=True, pos=nx.spring_layout(graph))
        plt.show()  # check if same as in the doc visually
        f.savefig("input.pdf", bbox_inches='tight')

    n = graph.number_of_nodes()
    n_matrix, n_prime_matrix = _get_n_matrices(ajc_matrix, n)

    page_rank_matrix = np.array([1 / n for _ in range(n)])  # column page rank matrix
    page_rank_matrix.resize([n, 1])

    # 1 iteration
    k = 1
    pr_k_1 = np.matmul(linalg.matrix_power(n_matrix.T, k), page_rank_matrix)

    zero_pr = np.where(pr_k_1 == 0)[0]
    # zeros -> 6, 11, 12, 17

    # 2 iterations
    k = 2
    pr_k_2 = np.matmul(linalg.matrix_power(n_matrix.T, k), page_rank_matrix)

    # 3 iterations
    k = 3
    pr_k_3 = np.matmul(linalg.matrix_power(n_matrix.T, k), page_rank_matrix)

    pr_matrix = pr_until_conv(page_rank_matrix, n_matrix)
    pr_matrix_scaled = pr_until_conv(page_rank_matrix, n_prime_matrix)

    print("Fin.")


if __name__ == '__main__':
    s = 0.9  # scaling factor
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
    page_rank_analysis(ajc_matrix, s)
