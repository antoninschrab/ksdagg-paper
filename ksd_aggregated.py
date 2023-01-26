"""
Functions for computing KSDAgg test for a collection of kernels
using either a wild bootstrap or a parametic bootstrap.
"""

from kernel import stein_kernel_matrices, compute_median_bandwidth, compute_ksd
import numpy as np


def ksdagg_wild(
    seed,
    X,
    score_X,
    alpha,
    beta_imq,
    kernel_type,
    weights_type,
    l_minus,
    l_plus,
    B1,
    B2,
    B3,
):
    """
    Compute KSDAgg using a wild bootstrap using bandwidths
    2 ** i * median_bandwidth for i = l_minus,...,l_plus.
    inputs: seed: non-negative integer
            X: (m,d) array of samples (m d-dimensional points)
            score_X: (m,d) array of score values for X
            alpha: real number in (0,1) (level of the test)
            beta_imq: parameter beta in (0,1) for the IMQ kernel
            kernel_type: "imq"
            weights_type: "uniform", "decreasing", "increasing" or "centred"
                see Section 5.1 of MMD Aggregated Two-Sample Test (Schrab et al., 2021)
            l_minus: integer for bandwidth collection
            l_plus: integer for bandwidth collection
            B1: number of simulated test statistics to estimate the quantiles
            B2: number of simulated test statistics to estimate the level
            B3: number of iterations for the bisection method
    output: result of KSDAgg (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    m = X.shape[0]
    assert m >= 2
    assert 0 < alpha and alpha < 1
    assert l_minus <= l_plus
    median_bandwidth = compute_median_bandwidth(seed, X)
    bandwidths_collection = np.array(
        [2**i * median_bandwidth for i in range(l_minus, l_plus + 1)]
    )
    N = 1 + l_plus - l_minus
    weights = create_weights(N, weights_type)
    stein_kernel_matrices_list = stein_kernel_matrices(
        X, score_X, kernel_type, bandwidths_collection, beta_imq
    )
    return ksdagg_wild_custom(
        seed,
        stein_kernel_matrices_list,
        weights,
        alpha,
        B1,
        B2,
        B3,
    )


def ksdagg_wild_custom(seed, stein_kernel_matrices_list, weights, alpha, B1, B2, B3):
    """
    Compute KSDAgg using a wild bootstrap with custom kernel matrices and weights.
    inputs: seed: non-negative integer
            stein_kernel_matrices_list: list of N stein kernel matrices
            weights: (N,) array consisting of positive entries summing to 1
            alpha: real number in (0,1) (level of the test)
            B1: number of simulated test statistics to estimate the quantiles
            B2: number of simulated test statistics to estimate the level
            B3: number of iterations for the bisection method
    output: result of KSDAgg (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    m = stein_kernel_matrices_list[0].shape[0]
    N = len(stein_kernel_matrices_list)
    assert len(stein_kernel_matrices_list) == weights.shape[0]
    assert m >= 2
    assert 0 < alpha and alpha < 1

    # Step 1: compute all simulated KSD estimates efficiently
    M = np.zeros((N, B1 + B2 + 1))
    rs = np.random.RandomState(seed)
    R = rs.choice([1.0, -1.0], size=(B1 + B2 + 1, m))  # (B1+B2+1, m) Rademacher
    R[B1] = np.ones(m)
    R = R.transpose()  # (m, B1+B2+1)
    for i in range(N):
        H = stein_kernel_matrices_list[i]
        np.fill_diagonal(H, 0)
        # (B1+B2+1, ) wild bootstrap KSD estimates
        M[i] = np.sum(R * (H @ R), 0) / (m * (m - 1))
    KSD_original = M[:, B1]
    M1_sorted = np.sort(M[:, :B1])  # (N, B1)
    M2 = M[:, B1 + 1 :]  # (N, B2)

    # Step 2: compute u_alpha using the bisection method
    quantiles = np.zeros((N, 1))  # (1-u*w_lambda)-quantiles for the N bandwidths
    u_min = 0
    u_max = np.min(1 / weights)
    for _ in range(B3):
        u = (u_max + u_min) / 2
        for i in range(N):
            quantiles[i] = M1_sorted[
                i, int(np.ceil(B1 * (1 - u * weights[i]))) - 1
            ]
        P_u = np.sum(np.max(M2 - quantiles, 0) > 0) / B2
        if P_u <= alpha:
            u_min = u
        else:
            u_max = u
    u = u_min

    # Step 3: output test result
    for i in range(N):
        if KSD_original[i] > M1_sorted[i, int(np.ceil(B1 * (1 - u * weights[i]))) - 1]:
            return 1
    return 0


def ksdagg_parametric(
    X,
    score_X,
    alpha,
    beta_imq,
    kernel_type,
    weights_type,
    l_minus,
    l_plus,
    bandwidth_reference,
    B1_parametric,
    B2_parametric,
    B3,
):
    """
    Compute KSDAgg using a parametric bootstrap using bandwidths
    2 ** i * median_bandwidth for i = l_minus,...,l_plus.
    inputs: seed: non-negative integer
            X: (m,d) array of samples (m d-dimensional points)
            score_X: (m,d) array of score values for X
            alpha: real number in (0,1) (level of the test)
            beta_imq: parameter beta in (0,1) for the IMQ kernel
            kernel_type: "imq"
            weights_type: "uniform", "decreasing", "increasing" or "centred"
                see Section 5.1 of MMD Aggregated Two-Sample Test (Schrab et al., 2021)
            l_minus: integer for bandwidth collection
            l_plus: integer for bandwidth collection
            bandwidth_reference: non-negative number
                (if 0 then median bandwidth is computed)
            B1_parametric: (N, B1) array of ksd values computed with N bandwidths
                using samples from the model
            B2_parametric: (N, B2) array of ksd values computed with N bandwidths
                using samples from the model
            B3: number of iterations for the bisection method
    output: result of KSDAgg (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    assert bandwidth_reference >= 0
    if bandwidth_reference == 0:
        bandwidth_reference = compute_median_bandwidth(seed=0, X=X)
    bandwidths_collection = np.array(
        [2**i * bandwidth_reference for i in range(l_minus, l_plus + 1)]
    )
    N = bandwidths_collection.shape[0]  # N = 1 + l_plus - l_minus
    weights = create_weights(N, weights_type)
    ksd_values = compute_ksd(
        X,
        score_X,
        kernel_type,
        bandwidths_collection,
        beta_imq,
    )
    return ksdagg_parametric_custom(
        ksd_values,
        alpha,
        weights,
        B1_parametric,
        B2_parametric,
        B3,
    )


def ksdagg_parametric_custom(
    ksd_values,
    alpha,
    weights,
    B1_parametric,
    B2_parametric,
    B3,
):
    """
    Compute KSDAgg using a parametric bootstrap with custom kernel matrices and weights.
    inputs: ksd_values: (N,) array consisting of KSD values
                for N bandwidths for inputs X and score_X
            alpha: real number in (0,1) (level of the test)
            weights: (N,) array consisting of positive entries summing to 1
            B1_parametric: (N, B1) array of ksd values computed with N bandwidths
                using samples from the model
            B2_parametric: (N, B2) array of ksd values computed with N bandwidths
                using samples from the model
            B3: number of iterations for the bisection method
    output: result of KSDAgg (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    B1 = B1_parametric.shape[1]
    B2 = B2_parametric.shape[1]
    N = ksd_values.shape[0]
    quantiles = np.zeros((N, 1))  # (1-u*w_lambda)-quantiles for the N bandwidths
    u_min = 0
    u_max = np.min(1 / weights)
    for _ in range(B3):
        u = (u_max + u_min) / 2
        for i in range(N):
            quantiles[i] = B1_parametric[i, int(np.ceil(B1 * (1 - u * weights[i]))) - 1]
        P_u = np.mean(np.max(B2_parametric - quantiles, 0) > 0)
        if P_u <= alpha:
            u_min = u
        else:
            u_max = u
    u = u_min
    for i in range(N):
        if (
            ksd_values[i]
            > B1_parametric[i, int(np.ceil(B1 * (1 - u * weights[i]))) - 1]
        ):
            return 1
    return 0


def create_weights(N, weights_type):
    """
    Create weights as defined in Section 5.1 of MMD Aggregated Two-Sample Test (Schrab et al., 2021).
    inputs: N: number of bandwidths to test
            weights_type: "uniform" or "decreasing" or "increasing" or "centred"
    output: (N,) array of weights
    """
    if weights_type == "uniform":
        weights = np.array(
            [
                1 / N,
            ]
            * N
        )
    elif weights_type == "decreasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / (i * normaliser) for i in range(1, N + 1)])
    elif weights_type == "increasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / ((N + 1 - i) * normaliser) for i in range(1, N + 1)])
    elif weights_type == "centred":
        if N % 2 == 1:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 1) for i in range(1, N + 1)])
            weights = np.array(
                [1 / ((abs((N + 1) / 2 - i) + 1) * normaliser) for i in range(1, N + 1)]
            )
        else:
            normaliser = sum(
                [1 / (abs((N + 1) / 2 - i) + 0.5) for i in range(1, N + 1)]
            )
            weights = np.array(
                [
                    1 / ((abs((N + 1) / 2 - i) + 0.5) * normaliser)
                    for i in range(1, N + 1)
                ]
            )
    else:
        raise ValueError(
            'The value of weights_type should be "uniform" or'
            '"decreasing" or "increasing" or "centred".'
        )
    return weights
