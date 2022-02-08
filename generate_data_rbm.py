"""
We generate data for the Gaussian-Bernoulli Restricted Boltzmann 
Machine experiment as proposed in Section 4.4 of our paper
KSD Aggregated Goodness-of-fit Test
Antonin Schrab, Benjamin Guedj, Arthur Gretton
https://arxiv.org/pdf/2202.00824.pdf
The data is saved in the directory data/RBM.
"""

import kgof.density as density
import kgof.data as data
from pathlib import Path
import numpy as np
import time


def rbm_samples_scores(
    seed,
    m,
    sigma,
    dx=50,
    dh=40,
    burnin_number=2000,
):
    """
    Generate data for the Gaussian-Bernoulli Restricted Boltzmann Machine (RBM) experiment.
    The entries of the matrix B are perturbed.
    This experiment was first proposed by Liu et al., 2016 (Section 6)
    inputs: seed: non-negative integer
            m: number of samples
            sigma: standard deviation of Gaussian noise
            dx: dimension of observed output variable
            dh: dimension of binary latent variable
            burnin_number: number of burn-in iterations for Gibbs sampler
    outputs: 2-tuple consisting of
            (m,dx) array of samples generated using the perturbed RBM
            (m,dx) array of scores computed using the non-perturbed RBM (model)
    """
    # the perturbed model is fixed, randomness comes from sampling
    rs = np.random.RandomState(0)

    # Model p
    B = rs.randint(0, 2, (dx, dh)) * 2 - 1.0
    b = rs.randn(dx)
    c = rs.randn(dh)
    p = density.GaussBernRBM(B, b, c)

    # Sample from q
    B_perturbed = B + rs.randn(dx, dh) * sigma
    q = density.GaussBernRBM(B_perturbed, b, c)
    ds = q.get_datasource()
    ds.burnin = burnin_number
    samples = ds.sample(m, seed=seed).data()

    # Compute score under p
    scores = p.grad_log(samples)

    return samples, scores


Path("data/RBM").mkdir(exist_ok=True, parents=True)
repetitions = 200
m = 1000
d = 50
dh = 40
sigmas = [0, 0.01, 0.02, 0.03, 0.04]
seed_count = 0
print("Starting")
t = time.time()
for s in range(len(sigmas)):
    sigma = sigmas[s]
    X = np.empty((repetitions, m, d))
    score_X = np.empty((repetitions, m, d))
    for r in range(repetitions):
        seed_count += 1
        X[r], score_X[r] = rbm_samples_scores(seed_count, m, sigma, d, dh)
        if (r + 1) % 10 == 0:
            print(
                "Step s =",
                s + 1,
                "/",
                len(sigmas),
                ",",
                r + 1,
                "/",
                repetitions,
                "time:",
                time.time() - t,
            )
            t = time.time()
    np.save("data/RBM/X_rbm_s" + str(int(sigma * 100)) + ".npy", X)
    np.save("data/RBM/score_X_rbm_s" + str(int(sigma * 100)) + ".npy", score_X)
print("RBM data has been saved in data/RBM.")
