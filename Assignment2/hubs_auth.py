"""
Description : Helps compute Hubs and Authority scores using the Easley and Kleinberg Hub and Authority Score formulations
Course : CSCI 5760
Author : Paul Louis <paul.louis@ontariotechu.net>
Usage : python3.9 hubs_auth.py
"""

from math import sqrt  # used to calculate the square root of a number
import fractions  # used to print a variable as a fraction instead of as a decimal
import numpy as np  # used for matrix arithmetic
from numpy.linalg import linalg  # used for matrix power arithmetic


def _run_auth_updates(ajc_matrix, hubs_init, k):
    # Auth update is given by the formula: h^k = (A.Transpose X A)^k-1 X A.Transpose X h_init
    interim = np.matmul(ajc_matrix.T, ajc_matrix)  # calculate (A.Transpose X A)
    interim = linalg.matrix_power(interim, k - 1)  # get k-1 th power of (A.Transpose X A)
    interim = np.matmul(interim, ajc_matrix.T)  # multiply (A X A.Transpose)^k-1 with A.Transpose
    auth_k = np.matmul(interim, hubs_init)  # finally multiply with initial hub matrix
    auth_k_norm = auth_k / np.sum(auth_k)  # normalize by dividing each auth score with sum of auth scores
    return auth_k, auth_k_norm


def _run_hub_updates(ajc_matrix, hubs_init, k):
    # Hub update is given by the formula: h^k = (A X A.Transpose) X h_init
    interim = np.matmul(ajc_matrix, ajc_matrix.T)  # calculate (A X A.Transpose)
    interim = linalg.matrix_power(interim, k)  # get kth power of (A X A.Transpose)
    hub_k = np.matmul(interim, hubs_init)  # multiply calculated value with initial hub matrix
    hub_k_norm = hub_k / np.sum(hub_k)  # normalize by dividing each hub score with sum of hub scores
    return hub_k, hub_k_norm


def run_hubs_auth(ajc_matrix, k):
    n = int(sqrt(ajc_matrix.size))  # get the number of nodes in the graph

    hubs_init = np.ones(shape=[n, 1])  # initialize the hub matrix with unit values

    # get the authority and normalized authority scores after k updates
    auth_k, auth_k_norm = _run_auth_updates(ajc_matrix, hubs_init, k)

    # get the hub and normalized hub scores after k updates
    hub_k, hub_k_norm = _run_hub_updates(ajc_matrix, hubs_init, k)

    return auth_k, auth_k_norm, hub_k, hub_k_norm


if __name__ == '__main__':
    # https://stackoverflow.com/a/42209716
    np.set_printoptions(
        formatter={'all': lambda x: str(fractions.Fraction(x).limit_denominator())})  # display the scores as fractions

    # answer to question 5.a
    k = 2  # number of updates
    # the adjacency matrix manually created for the network given
    ajc_matrix = np.array([[0, 0, 0, 0, 0, 0, 0],  # A
                           [0, 0, 0, 0, 0, 0, 0],  # B
                           [1, 0, 0, 0, 0, 0, 0],  # C
                           [1, 0, 0, 0, 0, 0, 0],  # D
                           [1, 1, 0, 0, 0, 0, 0],  # E
                           [0, 1, 0, 0, 0, 0, 1],  # F
                           [0, 0, 0, 0, 0, 0, 0]  # G
                           ])
    auth_k, auth_k_norm, hub_k, hub_k_norm = run_hubs_auth(ajc_matrix, k)
    print('---------------')
    print(auth_k)
    print(auth_k_norm)
    print(hub_k)
    print(hub_k_norm)
    print('---------------')

    # answer to question 5.b.1
    k = 2
    # adjacency matrix updated to include X, Y and edge (Y,X)
    ajc_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],  # A
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B
                           [1, 0, 0, 0, 0, 0, 0, 0, 0],  # C
                           [1, 0, 0, 0, 0, 0, 0, 0, 0],  # D
                           [1, 1, 0, 0, 0, 0, 0, 0, 0],  # E
                           [0, 1, 0, 0, 0, 0, 1, 0, 0],  # F
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],  # G
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],  # X
                           [0, 0, 0, 0, 0, 0, 0, 1, 0]  # Y
                           ])
    auth_k, auth_k_norm, _, _ = run_hubs_auth(ajc_matrix, k)

    print('---------------')
    print(auth_k)
    print(auth_k_norm)
    print('---------------')

    # answer to question 5.b.2
    k = 2
    # adjacency matrix updated to include X, Y and edge (Y,X)
    ajc_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],  # A
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B
                           [1, 0, 0, 0, 0, 0, 0, 0, 0],  # C
                           [1, 0, 0, 0, 0, 0, 0, 0, 0],  # D
                           [1, 1, 0, 0, 0, 0, 0, 0, 0],  # E
                           [0, 1, 0, 0, 0, 0, 1, 0, 0],  # F
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],  # G
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],  # X
                           [1, 0, 0, 0, 0, 0, 0, 1, 0]   # Y
                           ])
    auth_k, auth_k_norm, _, _ = run_hubs_auth(ajc_matrix, k)

    print('---------------')
    print(auth_k)
    print(auth_k_norm)
    print('---------------')

    # answer to question 5.b.3
    k = 2
    # adjacency matrix updated to include X, Y and edge (Y,X)
    ajc_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],  # A
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B
                           [1, 0, 0, 0, 0, 0, 0, 0, 0],  # C
                           [1, 0, 0, 0, 0, 0, 0, 0, 0],  # D
                           [1, 1, 0, 0, 0, 0, 0, 0, 0],  # E
                           [0, 1, 0, 0, 0, 0, 1, 0, 0],  # F
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],  # G
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],  # X
                           [1, 1, 0, 0, 0, 0, 1, 1, 0]   # Y
                           ])
    auth_k, auth_k_norm, _, _ = run_hubs_auth(ajc_matrix, k)

    print('---------------')
    print(auth_k)
    print(auth_k_norm)
    print('---------------')

    print("Fin.")
