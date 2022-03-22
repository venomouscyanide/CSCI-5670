from math import sqrt
import fractions
import numpy as np
from numpy.linalg import linalg


def run_hubs_auth(ajc_matrix, k):
    n = int(sqrt(ajc_matrix.size))

    auth_init = np.ones(shape=[n, 1])
    hubs_init = np.ones(shape=[n, 1])

    # auth update
    interim = np.matmul(ajc_matrix.T, ajc_matrix)
    interim = linalg.matrix_power(interim, k - 1)
    interim = np.matmul(interim, ajc_matrix.T)
    auth_k = np.matmul(interim, hubs_init)
    auth_k_norm = auth_k / np.sum(auth_k)

    # hubs update
    interim = np.matmul(ajc_matrix, ajc_matrix.T)
    interim = linalg.matrix_power(interim, k)
    hub_k = np.matmul(interim, hubs_init)
    hub_k_norm = hub_k / np.sum(hub_k)

    # https://stackoverflow.com/a/42209716
    np.set_printoptions(
        formatter={'all': lambda x: str(fractions.Fraction(x).limit_denominator())})
    print("Fin.")


if __name__ == '__main__':
    k = 2  # number of updates

    ajc_matrix = np.array([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0],
                           [1, 1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 1]])
    run_hubs_auth(ajc_matrix, k)
