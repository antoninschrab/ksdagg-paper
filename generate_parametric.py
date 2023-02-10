"""
We compute KSD values for the parametric bootstrap for the 
three experiments proposed in Sections 4.3, 4.4 & 4.5 of our paper
KSD Aggregated Goodness-of-fit Test
Antonin Schrab, Benjamin Guedj, Arthur Gretton
https://arxiv.org/pdf/2202.00824.pdf
The KSD values are saved in the directory parametric/.
"""

from kernel import compute_ksd, compute_median_bandwidth
from pathlib import Path
import numpy as np
import time
from ksdagg import create_weights
import scipy.spatial


def generate_parametric(
    X_rep,
    score_X_rep,
    B,
    B1,
    B2,
    kernel_type,
    l_minus,
    l_plus,
    beta_imq,
    verbose=True,
):
    """
    Compute KSD values.
    inputs: X_rep: (r,m,d) array of r repetitions of m d-dimensional samples
            score_X_rep: (r,m,d) array of score values for X
            B: positive integer
            B1: positive integer
            B2: positive integer
            kernel_type: "imq"
            l_minus: integer for bandwidth collection
            l_plus: integer for bandwidth collection
            beta_imq: parameter beta in (0,1) for the IMQ kernel
            verbose: boolean (print statements)
    output: 4-tuple with ordered elements:
            B_parametric: (B,) array of KSD values computed with m samples
            B1_parametric: (N,B1) array of KSD values computed with m samples for N bandwidths
            B1_parametric_split: (N,B1) array of KSD values computed with m/2 samples for N bandwidths
            B2_parametric: (N,B2) array of KSD values computed with m samples for N bandwidths
            median_bandwidth: float
    """
    assert X_rep.shape[0] >= B1 + B2
    m = X_rep.shape[1]

    # compute median bandwidth
    median_bandwidth = compute_median_bandwidth(0, X_rep[0])

    # define bandwidth_multipliers and weights
    bandwidth_multipliers = np.array([2**i for i in range(l_minus, l_plus + 1)])
    bandwidths_collection = np.array(
        [b * median_bandwidth for b in bandwidth_multipliers]
    )
    N = bandwidth_multipliers.shape[0]  # N = 1 + l_plus - l_minus

    # B_parametric
    t = time.time()
    B_parametric = np.zeros((B,))
    for b in range(B):
        if (b + 1) % 25 == 0 and verbose:
            print("Step: 1 / 4,", b + 1, "/", B, time.time() - t)
            t = time.time()
        B_parametric[b] = compute_ksd(
            X_rep[b],
            score_X_rep[b],
            kernel_type,
            np.array([median_bandwidth]),
            beta_imq,
        )
    B_parametric = np.sort(B_parametric.T)

    B1_parametric = np.zeros((B1, N))
    for b in range(B1):
        if (b + 1) % 25 == 0 and verbose:
            print("Step: 2 / 4,", b + 1, "/", B1, time.time() - t)
            t = time.time()
        B1_parametric[b] = compute_ksd(
            X_rep[b], score_X_rep[b], kernel_type, bandwidths_collection, beta_imq
        )
    B1_parametric = np.sort(B1_parametric.T)

    B2_parametric = np.zeros((B2, N))
    for b in range(B2):
        if (b + 1) % 25 == 0 and verbose:
            print("Step: 3 / 4,", b + 1, "/", B2, time.time() - t)
            t = time.time()
        B2_parametric[b] = compute_ksd(
            X_rep[B1 + b],
            score_X_rep[B1 + b],
            kernel_type,
            bandwidths_collection,
            beta_imq,
        )
    B2_parametric = B2_parametric.T

    B1_parametric_split = np.zeros((B1, N))
    split_size = int(m // 2)
    for b in range(B1):
        if (b + 1) % 25 == 0 and verbose:
            print("Step: 4 / 4,", b + 1, "/", B1, time.time() - t)
            t = time.time()
        B1_parametric_split[b] = compute_ksd(
            X_rep[b][:split_size],
            score_X_rep[b][:split_size],
            kernel_type,
            bandwidths_collection,
            beta_imq,
        )
    B1_parametric_split = np.sort(B1_parametric_split.T)

    return (
        B_parametric,
        B1_parametric,
        B1_parametric_split,
        B2_parametric,
        median_bandwidth,
    )


# Gamma
def score_gamma(x, k, theta):
    """
    Compute score function of one-dimensional Gamma distribution.
    inputs: x: real number at which the score function is evaluated
            k: positive number (shape parameter of Gamma distribution)
            theta: positive number (scale parameter of Gamma distribution)
    output: score
    """
    return (k - 1) / x - 1 / theta


l_minus = 0
l_plus = 10
rs = np.random.RandomState(0)
number_samples = 500
k_p = 5
theta_p = 5
B = 500
B1 = 500
B2 = 500
X_rep_param = rs.gamma(k_p, theta_p, (B1 + B2, number_samples, 1))
score_X_rep_param = score_gamma(X_rep_param, k_p, theta_p)
kernel_type = "imq"
beta_imq = 0.5
print("Starting Gamma")
(
    B_parametric,
    B1_parametric,
    B1_parametric_split,
    B2_parametric,
    median_bandwidth,
) = generate_parametric(
    X_rep_param,
    score_X_rep_param,
    B,
    B1,
    B2,
    kernel_type,
    l_minus,
    l_plus,
    beta_imq,
    verbose=True,
)
Path("parametric/Gamma").mkdir(exist_ok=True, parents=True)
np.save("parametric/Gamma/B_parametric" + str(number_samples) + ".npy", B_parametric)
np.save("parametric/Gamma/B1_parametric" + str(number_samples) + ".npy", B1_parametric)
np.save(
    "parametric/Gamma/B1_parametric_split" + str(number_samples) + ".npy",
    B1_parametric_split,
)
np.save("parametric/Gamma/B2_parametric" + str(number_samples) + ".npy", B2_parametric)
np.save("parametric/Gamma/bandwidth" + str(number_samples) + ".npy", median_bandwidth)
print("Gamma parametric has been saved in parametric/Gamma/.")


# Gaussian-Bernoulli Restricted Boltzmann Machine
l_minus = -20
l_plus = 0
rs = np.random.RandomState(0)
number_samples = 1000
d = 50
X_rep_all = np.load("data/RBM/X_rbm_s0.npy").reshape(-1, d)
score_X_rep_all = np.load("data/RBM/score_X_rbm_s0.npy").reshape(-1, d)
B = 500
B1 = 500
B2 = 500
X_rep_param = np.zeros((B1 + B2, number_samples, d))
score_X_rep_param = np.zeros((B1 + B2, number_samples, d))
print("Starting RBM")
for i in range(B1 + B2):
    indices = rs.choice(X_rep_all.shape[0], size=number_samples, replace=False)
    X_rep_param[i] = X_rep_all[indices]
    score_X_rep_param[i] = score_X_rep_all[indices]
kernel_type = "imq"
beta_imq = 0.5
(
    B_parametric,
    B1_parametric,
    B1_parametric_split,
    B2_parametric,
    median_bandwidth,
) = generate_parametric(
    X_rep_param,
    score_X_rep_param,
    B,
    B1,
    B2,
    kernel_type,
    l_minus,
    l_plus,
    beta_imq,
    verbose=True,
)
Path("parametric/RBM").mkdir(exist_ok=True, parents=True)
np.save("parametric/RBM/B_parametric" + str(number_samples) + ".npy", B_parametric)
np.save("parametric/RBM/B1_parametric" + str(number_samples) + ".npy", B1_parametric)
np.save(
    "parametric/RBM/B1_parametric_split" + str(number_samples) + ".npy",
    B1_parametric_split,
)
np.save("parametric/RBM/B2_parametric" + str(number_samples) + ".npy", B2_parametric)
np.save("parametric/RBM/bandwidth" + str(number_samples) + ".npy", median_bandwidth)
print("RBM parametric has been saved in parametric/RBM/.")


# Normalizing Flow (load data)
d = 28 ** 2
X_rep_all = np.load("data/NF_MNIST/bootstrap/X_mnist_level.npy").reshape(-1, d)
score_X_rep_all = np.load(
    "data/NF_MNIST/bootstrap/score_X_mnist_level.npy"
).reshape(-1, d)

# Normalizing Flow (ksd_aggregated.py)
print("Starting NF MNIST 1/2")
for number_samples in [100, 200, 300, 400, 500]:
    l_minus = -20
    l_plus = 0
    rs = np.random.RandomState(0)
    B = 500
    B1 = 500
    B2 = 500
    B3 = 0
    X_rep_param = np.zeros((B1 + B2, number_samples, d))
    score_X_rep_param = np.zeros((B1 + B2, number_samples, d))
    for i in range(B1 + B2):
        indices = rs.choice(X_rep_all.shape[0], size=number_samples, replace=False)
        X_rep_param[i] = X_rep_all[indices]
        score_X_rep_param[i] = score_X_rep_all[indices]
    kernel_type = "imq"
    beta_imq = 0.5
    (
        B_parametric,
        B1_parametric,
        B1_parametric_split,
        B2_parametric,
        median_bandwidth,
    ) = generate_parametric(
        X_rep_param,
        score_X_rep_param,
        B,
        B1,
        B2,
        kernel_type,
        l_minus,
        l_plus,
        beta_imq,
        verbose=True,
    )
    Path("parametric/NF_MNIST").mkdir(exist_ok=True, parents=True)
    np.save(
        "parametric/NF_MNIST/B_parametric" + str(number_samples) + ".npy", B_parametric
    )
    np.save(
        "parametric/NF_MNIST/B1_parametric" + str(number_samples) + ".npy",
        B1_parametric,
    )
    np.save(
        "parametric/NF_MNIST/B1_parametric_split" + str(number_samples) + ".npy",
        B1_parametric_split,
    )
    np.save(
        "parametric/NF_MNIST/B2_parametric" + str(number_samples) + ".npy",
        B2_parametric,
    )
    np.save(
        "parametric/NF_MNIST/bandwidth" + str(number_samples) + ".npy", median_bandwidth
    )
print("NF MNIST parametric has been saved in parametric/NF_MNIST/.")

# Definitions

def generate_parametric_2(
    X_rep_all,
    score_X_rep_all,
    B1,
    B2,
    sample_size,
    bandwidths,
    kernel="imq",
    weights_type="uniform", 
):
    """
    Compute KSD values.
    inputs: X_rep: (r,m,d) array of r repetitions of m d-dimensional samples
            score_X_rep: (r,m,d) array of score values for X
            B1: positive integer
            B2: positive integer
            sample_size: positive integer
            kernel_type: "imq"
            weights_type: "uniform", "centred", "increasing", "decreasing"
            
    output: 2-tuple with ordered elements:
            B1_parametric: (N, B1) array of KSD values computed with m samples for N bandwidths
            B2_parametric: (N, B2) array of KSD values computed with m samples for N bandwidths
    """
    number_bandwidths = len(bandwidths)
    
    # B1 & B2 parametric
    results = []
    for b in range(B1 + B2):
        rs = np.random.RandomState(b)
        indices = rs.choice(X_rep_all.shape[0], size=sample_size, replace=False)
        X_rep = X_rep_all[indices]
        score_X_rep = score_X_rep_all[indices]
        X = X_rep
        score_X = score_X_rep
        results.append(
            compute_ksd_2(
                X,
                score_X,
                bandwidths,
            )
        )
    
    B1_parametric = np.zeros((B1, number_bandwidths))
    for b in range(B1):
        B1_parametric[b] = results[b]
    B1_parametric = B1_parametric.T
    
    B2_parametric = np.zeros((B2, number_bandwidths))
    for b in range(B2):
        B2_parametric[b] = results[B1 + b]
    B2_parametric = B2_parametric.T
    
    return (
        B1_parametric,
        B2_parametric,
    )


def compute_ksd_2(X, score_X, bandwidths, weights_type="uniform"):
    m = X.shape[0]
    number_bandwidths = len(bandwidths)
    weights = create_weights(number_bandwidths, weights_type)
    beta_imq = 0.5
    p = X.shape[1]
    norms = np.sum(X ** 2, -1)
    dists = -2 * X @ X.T + np.expand_dims(norms, 1) + np.expand_dims(norms, 0)
    diffs = np.expand_dims(np.sum(X * score_X, -1), 1) - (X @ score_X.T)
    diffs = diffs + diffs.T
    KSD_original = np.zeros((number_bandwidths, ))
    for i in range(number_bandwidths):
        bandwidth = bandwidths[i]
        g = 1 / bandwidth ** 2
        res = 1 + g * dists
        kxy = res ** (-beta_imq)
        dkxy = 2 * beta_imq * g * (res) ** (-beta_imq - 1) * diffs
        d2kxy = 2 * (
            beta_imq * g * (res) ** (-beta_imq - 1) * p
            - 2
            * beta_imq
            * (beta_imq + 1)
            * g ** 2
            * dists
            * res ** (-beta_imq - 2)
        )
        H = score_X @ score_X.T * kxy + dkxy + d2kxy
        np.fill_diagonal(H, 0)
        r = np.ones(m)
        KSD_original[i] = r @ H @ r  / (m * (m - 1))
    return KSD_original

# Normalizing Flow (ksdagg.py)
print("Starting NF MNIST 2/2")
max_samples = 500
distances = scipy.spatial.distance.pdist(X_rep_all[:max_samples], "euclidean")  
distances = distances[distances > 0]
lambda_min = 1 
lambda_max = np.maximum(np.max(distances), 2)
number_bandwidths = 10
power = (lambda_max / lambda_min) ** (1 / (number_bandwidths - 1))
bandwidths = np.array([power ** i * lambda_min / X_rep_all.shape[1] for i in range(number_bandwidths)])
np.save(
    "parametric/NF_MNIST/bandwidths_ksdagg.npy",
    bandwidths,
)
for sample_size in [100, 200, 300, 400, 500]:
    B1 = 500
    B2 = 500
    (
        B1_parametric,
        B2_parametric,
    ) = generate_parametric_2(
    X_rep_all,
    score_X_rep_all,
    B1,
    B2,
    sample_size,
    bandwidths,
    kernel="imq",
    weights_type="uniform", 
)

    np.save(
        "parametric/NF_MNIST/B1_parametric_ksdagg_" + str(sample_size) + ".npy",
        B1_parametric,
    )
    np.save(
        "parametric/NF_MNIST/B2_parametric_ksdagg_" + str(sample_size) + ".npy",
        B2_parametric,
    )
print("NF MNIST parametric has been saved in parametric/NF_MNIST/.")

