"""
Functions for computing: 
Stein kernel matrices, 
KSD values, 
ratio KSD to standard deviation,
median bandwidth.
"""

import numpy as np
import scipy.spatial


def stein_kernel_matrices(
    X,
    score_X,
    kernel_type,
    bandwidths_collection,
    beta_imq,
):
    """
    Compute Stein kernel matrices for several bandwidths.
    Function adapted from https://github.com/pierreablin/ksddescent/blob/main/ksddescent/kernels.py
    inputs: X: (m,d) array of samples (m d-dimensional points)
            score_X: (m,d) array of score values for X
            kernel_type: "imq"
            bandwidths_collection: (N,) array of bandwidths
            beta_imq: parameter beta in (0,1) for the IMQ kernel
    outputs: list of N stein kernel matrices computed with the N bandwidths
    """
    if kernel_type == "imq":
        p = X.shape[1]
        norms = np.sum(X**2, -1)
        dists = -2 * X @ X.T + np.expand_dims(norms, 1) + np.expand_dims(norms, 0)
        diffs = np.expand_dims(np.sum(X * score_X, -1), 1) - (X @ score_X.T)
        diffs = diffs + diffs.T
        output_list = []
        for bandwith in bandwidths_collection:
            g = 1 / bandwith**2
            res = 1 + g * dists
            kxy = res ** (-beta_imq)
            dkxy = 2 * beta_imq * g * (res) ** (-beta_imq - 1) * diffs
            d2kxy = 2 * (
                beta_imq * g * (res) ** (-beta_imq - 1) * p
                - 2
                * beta_imq
                * (beta_imq + 1)
                * g**2
                * dists
                * res ** (-beta_imq - 2)
            )
            h = score_X @ score_X.T * kxy + dkxy + d2kxy
            output_list.append(h)
        return output_list
    else:
        raise ValueError('The value of kernel_type should be "imq".')


def compute_ksd(
    X,
    score_X,
    kernel_type,
    bandwidths_collection,
    beta_imq,
):
    """
    Compute KSD values for several bandwidths.
    Function adapted from https://github.com/pierreablin/ksddescent/blob/main/ksddescent/kernels.py
    inputs: X: (m,d) array of samples (m d-dimensional points)
            score_X: (m,d) array of score values for X
            kernel_type: "imq"
            bandwidths_collection: (N,) array of bandwidths
            beta_imq: parameter beta in (0,1) for the IMQ kernel
    outputs: (N,) array of KSD values for the N bandwidths
    """
    if kernel_type == "imq":
        p = X.shape[1]
        norms = np.sum(X**2, -1)
        dists = -2 * X @ X.T + np.expand_dims(norms, 1) + np.expand_dims(norms, 0)
        diffs = np.expand_dims(np.sum(X * score_X, -1), 1) - (X @ score_X.T)
        diffs = diffs + diffs.T
        output_list = []
        N = bandwidths_collection.shape[0]
        ksd_values = np.zeros((N,))
        for b in range(N):
            bandwidth = bandwidths_collection[b]
            g = 1 / bandwidth**2
            res = 1 + g * dists
            kxy = res ** (-beta_imq)
            dkxy = 2 * beta_imq * g * (res) ** (-beta_imq - 1) * diffs
            d2kxy = 2 * (
                beta_imq * g * (res) ** (-beta_imq - 1) * p
                - 2
                * beta_imq
                * (beta_imq + 1)
                * g**2
                * dists
                * res ** (-beta_imq - 2)
            )
            H = score_X @ score_X.T * kxy + dkxy + d2kxy
            np.fill_diagonal(H, 0)
            m = H.shape[0]
            r = np.ones(m)
            ksd_values[b] = r @ H @ r / (m * (m - 1))
        return ksd_values
    else:
        raise ValueError('The value of kernel_type should be "imq".')


def ratio_ksd_stdev(H, regulariser=10 ** (-8)):
    """
    Compute the estimated ratio of the KSD to the asymptotic standard deviation under the alternative.
    The original MMD formulation is attributed to (Eq. 3):
        F. Liu, W. Xu, J. Lu, G. Zhang, A. Gretton, and D. J. Sutherland
        Learning deep kernels for non-parametric two-sample tests
        International Conference on Machine Learning, 2020
        http://proceedings.mlr.press/v119/liu20m/liu20m.pdf
    inputs: H: (m, m) stein kernel matrix WITH diagonal entries
               (np.fill_diagonal(H, 0) has not been applied)
            regulariser: small positive number (we use 10**(-8) as done by Liu et al.)
    output: estimate of the ratio of KSD^2 and of the standard deviation under H_a
    warning: this function mutates H by applying np.fill_diagonal(H, 0)
    """
    m = H.shape[0]

    # compute variance
    H_column_sum = np.sum(H, axis=1)

    var = (
        4 / m**3 * np.sum(H_column_sum**2)
        - 4 / m**4 * np.sum(H_column_sum) ** 2
        + regulariser
    )
    # we should obtain var > 0, if var <= 0 then we discard the corresponding
    # bandwidth by returning a large negative value so that we do not select
    # the corresponding bandwidth when selecting the maximum of the outputs
    # of ratio_mmd_stdev for bandwidths in the collection
    if not var > 0:
        raise ValueError("Variance is negative. Try using a larger regulariser.")
        # return -1e10

    # compute original KSD estimate
    np.fill_diagonal(H, 0)
    v = np.ones(m)
    ksd = v @ H @ v / (m * (m - 1))

    return ksd / np.sqrt(var)


def compute_median_bandwidth(seed, X, max_samples=1000, min_value=0.0001):
    """
    Compute the median L^2-distance between all the points in X using at
    most max_samples samples and using a minimum threshold value min_value.
    inputs: seed: non-negative integer
            X: (m,d) array of samples
            max_samples: number of samples used to compute the median (int or None)
    output: median bandwidth (float)
    """
    if max_samples != None:
        rs = np.random.RandomState(seed)
        pX = rs.choice(X.shape[0], min(max_samples, X.shape[0]), replace=False)
        median_bandwidth = np.median(scipy.spatial.distance.pdist(X[pX], "euclidean"))
    else:
        median_bandwidth = np.median(scipy.spatial.distance.pdist(X, "euclidean"))
    return np.maximum(median_bandwidth, min_value)
