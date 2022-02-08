"""
Functions for computing KSD test for a fixed kernel
using either a wild bootstrap or a parametic bootstrap.
"""

from kernel import stein_kernel_matrices, compute_median_bandwidth, compute_ksd
import numpy as np


def ksd_median_wild(
    seed, X, score_X, alpha, beta_imq, kernel_type, B1, bandwidth_multiplier=1
):
    """
    Compute KSD test using a wild bootstrap with the median heuristic as
    kernel bandwidth multiplied by bandwidth_multiplier.
    This KSD test has been proposed by
        Kacper Chwialkowski, Heiko Strathmann and Arthur Gretton
        A Kernel Test of Goodness of Fit
        ICML 2016
        http://proceedings.mlr.press/v48/chwialkowski16.pdf
    inputs: seed: non-negative integer
            X: (m,d) array of samples (m d-dimensional points)
            score_X: (m,d) array of score values for X
            alpha: real number in (0,1) (level of the test)
            beta_imq: parameter beta in (0,1) for the IMQ kernel
            kernel_type: "imq"
            B1: number of simulated test statistics to estimate the quantiles
            bandwidth_multiplier: multiplicative factor for the median bandwidth
    output: result of KSD test using the median heuristic as kernel bandwidth
            (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    m = X.shape[0]
    assert m >= 2
    assert 0 < alpha and alpha < 1
    median_bandwidth = compute_median_bandwidth(seed, X)
    H = stein_kernel_matrices(
        X,
        score_X,
        kernel_type,
        np.array([median_bandwidth * bandwidth_multiplier]),
        beta_imq,
    )[0]
    return ksd_wild_custom(
        seed,
        H,
        alpha,
        B1,
    )


def ksd_wild_custom(seed, H, alpha, B1):
    """
    Compute KSD test using a wild bootstrap with custom kernel matrix.
    inputs: seed: non-negative integer
            H: stein kernel matrix
            alpha: real number in (0,1) (level of the test)
            B1: number of simulated test statistics to estimate the quantiles
    output: result of KSD test (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    m = H.shape[0]
    np.fill_diagonal(H, 0)
    rs = np.random.RandomState(seed)
    R = rs.choice([1.0, -1.0], size=(B1 + 1, m))  # (B1+1, m) Rademacher
    R[B1] = np.ones(m)
    R = R.transpose()  # (m, B1+1)
    M1 = np.sum(R * (H @ R), 0) / (m * (m - 1))
    KSD_original = M1[B1]
    M1_sorted = np.sort(M1[:B1])  # (B1,)
    if KSD_original > M1_sorted[int(np.ceil(B1 * (1 - alpha))) - 1]:
        return 1
    return 0


def ksd_parametric(
    X, score_X, alpha, beta_imq, kernel_type, bandwidth_reference, B_parametric
):
    """
    Compute KSD test using a parametric bootstrap with a reference kernel bandwidth
    This KSD test has been proposed by
        Kacper Chwialkowski, Heiko Strathmann and Arthur Gretton
        A Kernel Test of Goodness of Fit
        ICML 2016
        http://proceedings.mlr.press/v48/chwialkowski16.pdf
    inputs: seed: non-negative integer
            X: (m,d) array of samples (m d-dimensional points)
            score_X: (m,d) array of score values for X
            alpha: real number in (0,1) (level of the test)
            beta_imq: parameter beta in (0,1) for the IMQ kernel
            kernel_type: "imq"
            bandwidth_reference: non-negative number
                (if 0 then median bandwidth is computed)
            B_parametric: (N, B) array of ksd values computed with
                the reference bandwidth using samples from the model
    output: result of KSD test using the reference bandwidth
            (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    ksd_value = compute_ksd(
        X,
        score_X,
        kernel_type,
        np.array([bandwidth_reference]),
        beta_imq,
    )[0]
    return ksd_parametric_custom(ksd_value, alpha, B_parametric)


def ksd_parametric_custom(ksd_value, alpha, B_parametric):
    """
    Compute KSD test using a parametric bootstrap with kernel matrix
    inputs: ksd_values: (N,) array consisting of KSD values
                for N bandwidths for inputs X and score_X
            alpha: real number in (0,1) (level of the test)
            B_parametric: (N, B) array of ksd values computed with
                the reference bandwidth using samples from the model
    output: result of KSD test (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    B = B_parametric.shape[0]
    if ksd_value > B_parametric[int(np.ceil(B * (1 - alpha))) - 1]:
        return 1
    return 0
